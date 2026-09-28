use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use tile_kernel_builder_config::Settings;

use crate::target::{AscendTarget, FlagStyle, OutputFormat};

/// Configuration for a single kernel compilation.
pub struct CompileConfig {
    pub target: AscendTarget,
    pub output_format: OutputFormat,
    pub flag_style: FlagStyle,
    pub opt_level: u8,
    pub extra_includes: Vec<String>,
    pub extra_defines: Vec<String>,
    pub extra_libs: Vec<String>,
    pub validate: bool,
    pub auto_sync: bool,
}

impl CompileConfig {
    pub fn new(target: AscendTarget) -> Self {
        let flag_style = target.default_flag_style();
        CompileConfig {
            target,
            output_format: OutputFormat::Object,
            flag_style,
            opt_level: 2,
            extra_includes: Vec::new(),
            extra_defines: Vec::new(),
            extra_libs: Vec::new(),
            validate: true,
            auto_sync: true,
        }
    }
}

/// Build the bisheng command for the given configuration.
pub fn build_bisheng_command(
    settings: &Settings,
    input: &Path,
    output: &Path,
    config: &CompileConfig,
) -> Result<Command> {
    let bisheng = settings.cann_bisheng();
    if !bisheng.exists() {
        bail!("bisheng compiler not found at: {}", bisheng.display());
    }

    let mut cmd = Command::new(&bisheng);

    match config.flag_style {
        FlagStyle::CceAicore => {
            build_cce_aicore_flags(&mut cmd, settings, config);
        }
        FlagStyle::NpuArch => {
            build_npu_arch_flags(&mut cmd, config);
        }
    }

    // Preprocessor defines
    cmd.arg("-DTILING_KEY_VAR=0");
    for def in &config.extra_defines {
        cmd.arg(format!("-D{}", def));
    }

    // Extra include paths
    for inc in &config.extra_includes {
        cmd.arg(format!("-I{}", inc));
    }

    // Optimization and language standard
    cmd.arg(format!("-O{}", config.opt_level));
    cmd.arg("-std=c++17");

    // Output
    match config.output_format {
        OutputFormat::Object => {
            cmd.arg("-c").arg(input).arg("-o").arg(output);
        }
        OutputFormat::SharedLib => {
            cmd.arg("-fPIC")
                .arg("--shared")
                .arg(input)
                .arg("-o")
                .arg(output);
            for lib in &config.extra_libs {
                cmd.arg(format!("-l{}", lib));
            }
        }
    }

    Ok(cmd)
}

/// CceAicore flag path — used for 310P, 310B, 310, and 910B (C++ codegen).
fn build_cce_aicore_flags(cmd: &mut Command, settings: &Settings, config: &CompileConfig) {
    let aicore_arch = config.target.cce_aicore_arch();

    // Include paths for AscendC (tikcpp framework)
    for inc in settings.cann_bisheng_include_paths() {
        cmd.arg(format!("-I{}", inc.display()));
    }

    // CANN version header
    let version_header = settings.cann_version_header();
    if version_header.exists() {
        cmd.arg("-include").arg(&version_header);
    }

    // CCE aicore flags
    cmd.arg(format!("--cce-aicore-arch={}", aicore_arch));
    cmd.arg("--cce-aicore-only");
    cmd.arg("--cce-aicore-lang");
    cmd.arg("--cce-disable-kernel-global-attr-check");

    // Target-specific flags matching CANN cmake (bisheng_intf.cmake)
    match config.target {
        AscendTarget::Ascend910B1
        | AscendTarget::Ascend910B2
        | AscendTarget::Ascend910B3
        | AscendTarget::Ascend910B4 => {
            // c220 flags from CANN cmake — critical for cube engine operation
            cmd.arg("-mllvm").arg("-cce-aicore-stack-size=0x8000");
            cmd.arg("-mllvm")
                .arg("-cce-aicore-function-stack-size=0x8000");
            cmd.arg("-mllvm").arg("-cce-aicore-record-overflow=true");
            cmd.arg("-mllvm").arg("-cce-aicore-addr-transform");
            cmd.arg("-mllvm")
                .arg("-cce-aicore-dcci-insert-for-scalar=false");
        }
        AscendTarget::Ascend310P1 | AscendTarget::Ascend310P3 => {
            // m200 flags from CANN cmake
            cmd.arg("--cce-mask-opt");
            cmd.arg("-mllvm").arg("-cce-aicore-fp-ceiling=2");
            cmd.arg("-mllvm").arg("-cce-aicore-record-overflow=false");
            cmd.arg("-mllvm").arg("-cce-aicore-mask-opt=false");
        }
        _ => {
            // Conservative defaults for other targets
            cmd.arg("--cce-mask-opt");
            cmd.arg("-mllvm").arg("-cce-aicore-record-overflow=false");
        }
    }

    if config.auto_sync {
        cmd.arg("--cce-auto-sync");
    }
}

/// NpuArch flag path — used for 910B (TileLang-compatible).
fn build_npu_arch_flags(cmd: &mut Command, config: &CompileConfig) {
    let npu_arch = config
        .target
        .npu_arch()
        .expect("NpuArch flag style requires 910B target");

    cmd.arg(format!("--npu-arch={}", npu_arch));
    cmd.arg("-xasc");
}

/// Execute the bisheng command and return the result.
pub fn execute_bisheng(mut cmd: Command) -> Result<()> {
    // Clear LD_PRELOAD so that MLIR shims loaded by the codegen backend
    // do not leak into the compiler and cause symbol lookup failures.
    cmd.env_remove("LD_PRELOAD");
    let output = cmd.output().context("failed to execute bisheng")?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "bisheng compilation failed (exit={}):\n{}",
            output.status,
            stderr
        )
    }
}

/// Compile a cube kernel using AIC+AIV dual compilation, producing a `.so`.
///
/// This implements the full cube kernel pipeline:
/// 1. Dual compile: AIC (dav-c220-cube) + AIV (dav-c220-vec)
/// 2. Symbol rename: `_mix_aic`/`_mix_aiv` suffixes
/// 3. Link: ld.lld merge to device.o
/// 4. Generate + compile host stub with `.ascend.kernel` section
/// 5. Pack device binary via `ascendc_pack_kernel`
/// 6. Link into shared library
pub fn compile_cube_kernel(
    input: &Path,
    output_so: &Path,
    kernel_name: &str,
    config: &CompileConfig,
) -> Result<()> {
    let settings = Settings::new().map_err(|e| anyhow::anyhow!("settings: {}", e))?;

    let tmp_dir = tempfile::tempdir().context("create tmpdir")?;
    let tmp = tmp_dir.path();

    // --- Resolve tool paths ---
    let bisheng = settings.cann_bisheng();
    let cann = settings.cann();
    let lld = resolve_tool(cann, "tools/bisheng_compiler/bin/ld.lld", "ld.lld")?;
    let objcopy = resolve_tool(
        cann,
        "tools/bisheng_compiler/bin/llvm-objcopy",
        "llvm-objcopy",
    )?;
    let pack_kernel = resolve_pack_kernel(cann, settings.target_prefix())?;

    // --- Step 1: Dual compile AIC + AIV ---
    let aic_o = tmp.join("aic.o");
    let aiv_o = tmp.join("aiv.o");

    // Build common flags (tikcpp includes, defines, optimization)
    let common_flags = build_cube_common_flags(&settings, config);

    // AIC compilation (cube engine)
    {
        let mut cmd = Command::new(&bisheng);
        cmd.arg("-c");
        cmd.arg("--cce-aicore-arch=dav-c220-cube");
        for flag in &common_flags {
            cmd.arg(flag);
        }
        cmd.arg(input).arg("-o").arg(&aic_o);
        execute_bisheng(cmd).context("AIC (cube) compilation")?;
    }

    // AIV compilation (vector engine)
    {
        let mut cmd = Command::new(&bisheng);
        cmd.arg("-c");
        cmd.arg("--cce-aicore-arch=dav-c220-vec");
        cmd.arg("-mllvm")
            .arg("-cce-aicore-dcci-insert-for-scalar=false");
        for flag in &common_flags {
            cmd.arg(flag);
        }
        cmd.arg(input).arg("-o").arg(&aiv_o);
        execute_bisheng(cmd).context("AIV (vector) compilation")?;
    }

    // --- Step 2: Symbol rename to avoid duplicates ---
    rename_symbols(&objcopy, &aic_o, "_mix_aic")?;
    rename_symbols(&objcopy, &aiv_o, "_mix_aiv")?;

    // --- Step 3: Link AIC and AIV, then merge ---
    let device_aic_o = tmp.join("device_aic.o");
    let device_aiv_o = tmp.join("device_aiv.o");
    let device_o = tmp.join("device.o");

    // Link AIC relocatable
    run_cmd(
        Command::new(&lld)
            .arg("-m")
            .arg("aicorelinux")
            .arg("-r")
            .arg("-Ttext=0")
            .arg(&aic_o)
            .arg("-static")
            .arg("-o")
            .arg(&device_aic_o),
        "ld.lld AIC",
    )?;

    // Link AIV relocatable
    run_cmd(
        Command::new(&lld)
            .arg("-m")
            .arg("aicorelinux")
            .arg("-r")
            .arg("-Ttext=0")
            .arg(&aiv_o)
            .arg("-static")
            .arg("-o")
            .arg(&device_aiv_o),
        "ld.lld AIV",
    )?;

    // Final merge: EXEC binary
    run_cmd(
        Command::new(&lld)
            .arg("-x")
            .arg("-m")
            .arg("aicorelinux")
            .arg("-Ttext=0")
            .arg(&device_aic_o)
            .arg(&device_aiv_o)
            .arg("-static")
            .arg("-o")
            .arg(&device_o),
        "ld.lld merge",
    )?;

    let device_size = std::fs::metadata(&device_o).context("stat device.o")?.len() as usize;

    // --- Step 4: Generate host stub ---
    // Use lowercase SoC version: CANN's AscendCheckSoCVersion() and the
    // .ascend.kernel section names require lowercase (e.g. "ascend910b2").
    let soc = config.target.soc_version_lower();
    let soc_under = soc.replace('-', "_");
    let padded_size = (device_size + 3) / 4 * 4;

    let host_stub_cpp = tmp.join("host_stub.cpp");
    let host_stub_src = generate_host_stub(kernel_name, soc, &soc_under, padded_size, device_size);
    std::fs::write(&host_stub_cpp, &host_stub_src)?;

    // --- Step 5: Compile host stub ---
    let host_stub_o = tmp.join("host_stub.o");
    run_cmd(
        Command::new("gcc")
            .arg("-c")
            .arg("-fPIC")
            .arg("-O2")
            .arg("-std=c++17")
            .arg(format!("-I{}", cann.join("include").display()))
            .arg(&host_stub_cpp)
            .arg("-o")
            .arg(&host_stub_o),
        "gcc host_stub",
    )?;

    // --- Step 6: Pack device binary into host stub ---
    let packed_o = tmp.join("host_stub_packed.o");
    run_cmd(
        Command::new(&pack_kernel)
            .arg(&host_stub_o)
            .arg(&device_o)
            .arg("0") // kernel_type: 0=mix
            .arg(&packed_o),
        "ascendc_pack_kernel",
    )?;

    // --- Step 6b: Generate host-side _do launcher wrappers ---
    // Parse the input C++ to find kernel entry points and generate _do wrappers
    // that use runtime-level APIs (rtFunctionRegister + rtKernelLaunch).
    let host_wrappers_o = tmp.join("host_wrappers.o");
    {
        let input_src = std::fs::read_to_string(input)?;
        let host_src = generate_acl_do_wrappers(&input_src);
        if !host_src.is_empty() {
            let host_cpp = tmp.join("host_wrappers.cpp");
            std::fs::write(&host_cpp, &host_src)?;
            run_cmd(
                Command::new("gcc")
                    .arg("-c")
                    .arg("-fPIC")
                    .arg("-O2")
                    .arg("-std=c++17")
                    .arg(format!("-I{}", cann.join("include").display()))
                    .arg(&host_cpp)
                    .arg("-o")
                    .arg(&host_wrappers_o),
                "gcc host _do wrappers",
            )?;
        }
    }

    // --- Step 7: Link into shared library ---
    let runtime_a = resolve_runtime_lib(cann, settings.target_prefix())?;

    {
        let mut cmd = Command::new("gcc");
        cmd.arg("-shared")
            .arg("-fPIC")
            .arg("-o")
            .arg(output_so)
            .arg("-Wl,--whole-archive")
            .arg(&packed_o)
            .arg(&runtime_a)
            .arg("-Wl,--no-whole-archive");
        if host_wrappers_o.exists() {
            cmd.arg(&host_wrappers_o);
        }
        cmd.arg(format!("-L{}", cann.join("lib64").display()))
            .arg(format!(
                "-L{}",
                cann.join(settings.target_prefix()).join("lib64").display()
            ))
            .arg("-lascendcl")
            .arg("-lruntime")
            .arg("-lascend_dump")
            .arg("-lc_sec")
            .arg("-lstdc++")
            .arg("-lpthread");
        run_cmd(&mut cmd, "gcc shared lib")?;
    }

    Ok(())
}

/// Generate host-side `_do` launcher wrappers using runtime-level API.
///
/// The .so constructor uses `RegisterAscendBinary` which works at the runtime
/// level. We must use the matching runtime APIs (`rtFunctionRegister` +
/// `rtKernelLaunch`) to launch kernels — the ACL-level APIs
/// (`aclrtBinaryGetFunction`) are incompatible with this registration path.
///
/// Cube kernels on 910B expect args packed as:
///   [ffts_addr, gm0, gm1, ..., overflow_status]
fn generate_acl_do_wrappers(source: &str) -> String {
    let mut kernels = Vec::new();

    for line in source.lines() {
        if let Some(rest) = line.strip_prefix("extern \"C\" void ") {
            if let Some(do_pos) = rest.find("_do(") {
                let kernel_name = &rest[..do_pos];
                if let Some(paren_start) = rest.find('(') {
                    if let Some(paren_end) = rest.find(')') {
                        let params = &rest[paren_start + 1..paren_end];
                        let n_gm = params.split(", ").count() - 2; // skip blockDim, stream
                        kernels.push((kernel_name.to_string(), n_gm));
                    }
                }
            }
        }
    }

    if kernels.is_empty() {
        return String::new();
    }

    let mut out = String::with_capacity(4096);
    out.push_str("#include <stdint.h>\n");
    out.push_str("#include <stddef.h>\n\n");

    // Runtime-level function declarations (compatible with RegisterAscendBinary)
    out.push_str("extern \"C\" {\n");
    out.push_str("int32_t rtFunctionRegister(void *binHandle, const char *stubName, const char *stubName2, const char *funcName, uint32_t funcMode);\n");
    out.push_str("int32_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, void *smDesc, void *stream);\n");
    out.push_str("int aclrtMalloc(void **devPtr, size_t size, int policy);\n");
    out.push_str("int rtGetC2cCtrlAddr(void **addr, uint32_t *prefCnt);\n");
    out.push_str("extern void *g_kernel_handle;\n");
    out.push_str("}\n\n");

    // Lazy-init helpers for ffts_addr and overflow buffer
    out.push_str("static void *s_ffts_addr = nullptr;\n");
    out.push_str("static void *s_overflow = nullptr;\n\n");
    out.push_str("static void ensure_cube_ctx() {\n");
    out.push_str("    if (!s_ffts_addr) {\n");
    out.push_str("        uint32_t prefCnt = 0;\n");
    out.push_str("        rtGetC2cCtrlAddr(&s_ffts_addr, &prefCnt);\n");
    out.push_str("    }\n");
    out.push_str("    if (!s_overflow) {\n");
    out.push_str("        aclrtMalloc(&s_overflow, 8, 0);\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    // Each kernel needs a stable stub address for rtFunctionRegister/rtKernelLaunch.
    // We use a static char array per kernel as the stub pointer.
    for (kernel_name, n_gm) in &kernels {
        let total_args = n_gm + 2; // ffts + gm args + overflow

        // Static stub name string (used as both the name and the stub pointer)
        writeln!(
            out,
            "static const char s_stub_{}[] = \"{}\";",
            kernel_name, kernel_name
        )
        .unwrap();
        writeln!(out, "static bool s_registered_{} = false;\n", kernel_name).unwrap();

        let gm_params: Vec<String> = (0..*n_gm).map(|i| format!("uint8_t *gm{}", i)).collect();
        writeln!(
            out,
            "extern \"C\" void {}_do(uint32_t blockDim, void *stream, {}) {{",
            kernel_name,
            gm_params.join(", "),
        )
        .unwrap();
        writeln!(out, "    ensure_cube_ctx();").unwrap();

        // Register kernel function on first call
        writeln!(out, "    if (!s_registered_{}) {{", kernel_name).unwrap();
        writeln!(
            out,
            "        rtFunctionRegister(g_kernel_handle, s_stub_{0}, s_stub_{0}, s_stub_{0}, 0);",
            kernel_name
        )
        .unwrap();
        writeln!(out, "        s_registered_{} = true;", kernel_name).unwrap();
        writeln!(out, "    }}").unwrap();

        // Pack args: [ffts_addr, gm0, gm1, ..., overflow_status]
        writeln!(out, "    void *args[{}];", total_args).unwrap();
        writeln!(out, "    args[0] = s_ffts_addr;").unwrap();
        for i in 0..*n_gm {
            writeln!(out, "    args[{}] = (void*)gm{};", i + 1, i).unwrap();
        }
        writeln!(out, "    args[{}] = s_overflow;", n_gm + 1).unwrap();
        writeln!(
            out,
            "    rtKernelLaunch(s_stub_{}, blockDim, args, sizeof(args), nullptr, stream);",
            kernel_name
        )
        .unwrap();
        writeln!(out, "}}\n").unwrap();
    }

    out
}

/// Build the common bisheng flags for cube compilation (sans arch-specific flags).
fn build_cube_common_flags(settings: &Settings, config: &CompileConfig) -> Vec<String> {
    let mut flags = Vec::new();

    // Include paths for AscendC (tikcpp framework)
    for inc in settings.cann_bisheng_include_paths() {
        flags.push(format!("-I{}", inc.display()));
    }

    // CANN version header
    let version_header = settings.cann_version_header();
    if version_header.exists() {
        flags.push("-include".to_string());
        flags.push(version_header.display().to_string());
    }

    // Common CCE flags (from build_cube_kernel.sh)
    flags.push("--cce-aicore-only".to_string());
    if config.auto_sync {
        flags.push("--cce-auto-sync".to_string());
    }
    flags.push("--cce-mask-opt".to_string());
    flags.push("--cce-aicore-lang".to_string());
    flags.push("--cce-disable-kernel-global-attr-check".to_string());
    flags.push("-mllvm".to_string());
    flags.push("-cce-aicore-fp-ceiling=2".to_string());
    flags.push("-mllvm".to_string());
    flags.push("-cce-aicore-record-overflow=false".to_string());

    // Preprocessor defines
    flags.push("-DTILING_KEY_VAR=0".to_string());
    for def in &config.extra_defines {
        flags.push(format!("-D{}", def));
    }

    // Extra include paths
    for inc in &config.extra_includes {
        flags.push(format!("-I{}", inc));
    }

    // Optimization and language standard
    flags.push(format!("-O{}", config.opt_level));
    flags.push("-std=c++17".to_string());

    flags
}

/// Rename T/V symbols in an object file by appending a suffix.
fn rename_symbols(objcopy: &Path, obj: &Path, suffix: &str) -> Result<()> {
    // Get T and V symbols from the object
    let nm_output = Command::new("nm").arg(obj).output().context("run nm")?;

    if !nm_output.status.success() {
        // nm may fail on some objects; not fatal
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&nm_output.stdout);
    let syms: Vec<&str> = stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 && (parts[1] == "T" || parts[1] == "V") {
                let sym = parts[2];
                if !sym.starts_with('$') {
                    return Some(sym);
                }
            }
            None
        })
        .collect();

    for sym in syms {
        // Ignore errors on individual symbols
        let _ = Command::new(objcopy)
            .arg("--redefine-sym")
            .arg(format!("{}={}{}", sym, sym, suffix))
            .arg(obj)
            .output();
    }

    Ok(())
}

/// Generate C++ host stub source with `.ascend.kernel` ELF section.
fn generate_host_stub(
    kernel_name: &str,
    soc: &str,
    soc_under: &str,
    padded_size: usize,
    device_size: usize,
) -> String {
    let mut s = String::with_capacity(2048);

    writeln!(s, "#include <stdio.h>").unwrap();
    writeln!(s, "#include <stdint.h>").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "extern \"C\" {{").unwrap();
    writeln!(s, "uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);").unwrap();
    writeln!(s, "int UnregisterAscendBinary(void *hdl);").unwrap();
    writeln!(
        s,
        "bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);"
    )
    .unwrap();
    writeln!(s, "void AscendProfRegister();").unwrap();
    writeln!(s, "}}").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "void *g_kernel_handle = nullptr;").unwrap();
    writeln!(s, "static char ascendcErrMsg[1024] = {{0}};").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "struct ascend_kernels {{").unwrap();
    writeln!(s, "    uint32_t version;").unwrap();
    writeln!(s, "    uint32_t type_cnt;").unwrap();
    writeln!(s, "    uint32_t mix_type;").unwrap();
    writeln!(s, "    uint32_t mix_len;").unwrap();
    writeln!(s, "    uint32_t mix_file_len;").unwrap();
    writeln!(s, "    uint8_t mix_buf[{}];", padded_size).unwrap();
    writeln!(
        s,
        "}} __ascend_kernel_{soc_under}_{kernel_name} __attribute__ ((section (\".ascend.kernel.{soc_under}.{kernel_name}\"))) = {{"
    ).unwrap();
    writeln!(s, "    1, 1, 0, {}, {}, {{0}}", padded_size, device_size).unwrap();
    writeln!(s, "}};").unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "static void __register_kernels(void) __attribute__((constructor));"
    )
    .unwrap();
    writeln!(s, "void __register_kernels(void) {{").unwrap();
    writeln!(
        s,
        "    bool ok = AscendCheckSoCVersion(\"{}\", ascendcErrMsg);",
        soc
    )
    .unwrap();
    writeln!(s, "    uint32_t ret = RegisterAscendBinary(").unwrap();
    writeln!(
        s,
        "        (const char *)__ascend_kernel_{soc_under}_{kernel_name}.mix_buf,"
    )
    .unwrap();
    writeln!(
        s,
        "        __ascend_kernel_{soc_under}_{kernel_name}.mix_file_len,"
    )
    .unwrap();
    writeln!(s, "        0,").unwrap();
    writeln!(s, "        &g_kernel_handle);").unwrap();
    writeln!(s, "    AscendProfRegister();").unwrap();
    writeln!(s, "}}").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "class KernelHandleUnregister {{").unwrap();
    writeln!(s, "public:").unwrap();
    writeln!(s, "    static KernelHandleUnregister& GetInstance() {{ static KernelHandleUnregister inst; return inst; }}").unwrap();
    writeln!(s, "    ~KernelHandleUnregister() {{ if (g_kernel_handle) {{ UnregisterAscendBinary(g_kernel_handle); g_kernel_handle = nullptr; }} }}").unwrap();
    writeln!(s, "private:").unwrap();
    writeln!(s, "    KernelHandleUnregister() {{}}").unwrap();
    writeln!(s, "}};").unwrap();

    s
}

/// Resolve a CANN tool path, falling back to PATH lookup.
fn resolve_tool(cann: &Path, relative_path: &str, tool_name: &str) -> Result<PathBuf> {
    let cann_path = cann.join(relative_path);
    if cann_path.exists() {
        return Ok(cann_path);
    }
    // Try PATH
    let which_output = Command::new("which").arg(tool_name).output();
    if let Ok(output) = which_output {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return Ok(PathBuf::from(path));
        }
    }
    bail!(
        "{} not found at {} or in PATH",
        tool_name,
        cann_path.display()
    )
}

/// Resolve the ascendc_pack_kernel tool path.
fn resolve_pack_kernel(cann: &Path, target_prefix: &str) -> Result<PathBuf> {
    let path1 = cann.join(target_prefix).join("bin/ascendc_pack_kernel");
    if path1.exists() {
        return Ok(path1);
    }
    let path2 = cann.join("bin/ascendc_pack_kernel");
    if path2.exists() {
        return Ok(path2);
    }
    bail!(
        "ascendc_pack_kernel not found at {} or {}",
        path1.display(),
        path2.display()
    )
}

/// Resolve the libascendc_runtime.a path.
fn resolve_runtime_lib(cann: &Path, target_prefix: &str) -> Result<PathBuf> {
    let path1 = cann.join(target_prefix).join("lib64/libascendc_runtime.a");
    if path1.exists() {
        return Ok(path1);
    }
    let path2 = cann.join("lib64/libascendc_runtime.a");
    if path2.exists() {
        return Ok(path2);
    }
    bail!(
        "libascendc_runtime.a not found at {} or {}",
        path1.display(),
        path2.display()
    )
}

/// Run a command, returning an error with context on failure.
fn run_cmd(cmd: &mut Command, context: &str) -> Result<()> {
    // Clear LD_PRELOAD so that MLIR shims do not leak into compiler subprocesses.
    cmd.env_remove("LD_PRELOAD");
    let output = cmd
        .output()
        .with_context(|| format!("failed to execute {}", context))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("{} failed (exit={}):\n{}", context, output.status, stderr)
    }
}
