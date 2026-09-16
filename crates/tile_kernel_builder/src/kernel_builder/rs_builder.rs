use std::borrow::Borrow;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Deserialize;

use super::BuilderError;

const TARGET: &str = "davinci-huawei-none";

pub(super) fn build(kernel_path: impl AsRef<Path>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    build_with_env(kernel_path, None)
}

/// Variant that can override `TILERS_CODEGEN_PATH` on the cargo subprocess
/// without mutating the parent's environment — required for mixing PTO and
/// cpp kernel crates inside one workspace build.
pub(super) fn build_with_env(
    kernel_path: impl AsRef<Path>,
    codegen_path_override: Option<&str>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mlir_codegen = find_rustc_codegen_tile();

    let rustflags = vec![
        format!("-Zcodegen-backend={}", mlir_codegen.display()),
        "-Zcrate-attr=feature(register_tool)".into(),
        "-Zcrate-attr=register_tool(tile)".into(),
        "-Cpanic=abort".into(),
        "-Clto=off".into(),
    ];

    // Find the workspace root (where davinci-huawei-none.json lives)
    // by walking up from the kernel crate's Cargo.toml.
    let workspace_root = find_workspace_root(kernel_path.as_ref());

    // Resolve the target spec JSON file path. Using the full path avoids
    // reliance on RUST_TARGET_PATH which cargo may not forward to rustc.
    let target_arg = if let Some(ref root) = workspace_root {
        let json_path = root.join(format!("{TARGET}.json"));
        if json_path.is_file() {
            format!("--target={}", json_path.display())
        } else {
            format!("--target={TARGET}")
        }
    } else if let Ok(rtp) = env::var("RUST_TARGET_PATH") {
        let json_path = Path::new(&rtp).join(format!("{TARGET}.json"));
        if json_path.is_file() {
            format!("--target={}", json_path.display())
        } else {
            format!("--target={TARGET}")
        }
    } else {
        format!("--target={TARGET}")
    };

    let mut cargo = Command::new("cargo");
    cargo.args([
        "build",
        "--release",
        "--message-format=json-render-diagnostics",
        &target_arg,
    ]);

    let cargo_encoded_rustflags = join_checking_for_separators(rustflags, "\x1f");

    cargo
        .stderr(Stdio::inherit())
        .current_dir(kernel_path.as_ref())
        .env("CARGO_ENCODED_RUSTFLAGS", cargo_encoded_rustflags);

    if let Some(path) = codegen_path_override {
        cargo.env("TILERS_CODEGEN_PATH", path);
        // Force cargo to rebuild if the codegen path changes — without this,
        // a cached build from a previous `TILERS_CODEGEN_PATH=cpp` run would
        // get reused even after we switch to `pto`.
        println!("cargo:rerun-if-env-changed=TILERS_CODEGEN_PATH");
    }

    let build = cargo.output().expect("failed to execute cargo build");

    let stdout = String::from_utf8(build.stdout)?;

    if !build.status.success() {
        return Err(Box::new(BuilderError::BuildFailed));
    }

    // The codegen produces both .tile.o (device binary) and .tile.gen.cpp (C++ source).
    // We prefer the cmake/shared-library path (.so via C++) because it works in both
    // simulator and real NPU hardware. The .tile.o path (rtDevBinaryRegister) only
    // works on real hardware.
    let cpp_file = find_codegen_output(&stdout, "tile.gen.cpp");
    let tile_so = find_codegen_output(&stdout, "tile.so");

    // Cube kernels produce .tile.so directly (dual AIC+AIV compilation)
    if let Some(so_path) = tile_so {
        return Ok(so_path);
    }

    // Vector/scalar kernels: feed generated C++ through cmake to get libkernels.so
    if let Some(cpp_path) = cpp_file {
        return super::cpp_builder::build(cpp_path);
    }

    // Fallback: return .tile.o (legacy path)
    let tile_o = find_codegen_output(&stdout, "tile.o");
    tile_o.ok_or_else(|| Box::new(BuilderError::BuildFailed) as Box<dyn std::error::Error>)
}

fn join_checking_for_separators(strings: Vec<impl Borrow<str>>, sep: &str) -> String {
    for s in &strings {
        let s = s.borrow();
        assert!(!s.contains(sep), "{s:?} may not contain separator {sep:?}");
    }
    strings.join(sep)
}

// https://github.com/rust-lang/cargo/blob/1857880b5124580c4aeb4e8bc5f1198f491d61b1/src/cargo/util/paths.rs#L29-L52
fn dylib_path_envvar() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_FALLBACK_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}
fn dylib_path() -> Vec<PathBuf> {
    match env::var_os(dylib_path_envvar()) {
        Some(var) => env::split_paths(&var).collect(),
        None => Vec::new(),
    }
}

/// Walk up from `start` looking for the workspace root (directory containing
/// `davinci-huawei-none.json`).
fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    let mut dir = if start.is_dir() {
        start.canonicalize().ok()?
    } else {
        start.parent()?.canonicalize().ok()?
    };
    loop {
        if dir.join("davinci-huawei-none.json").is_file() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn find_rustc_codegen_tile() -> PathBuf {
    // Explicit override wins. The kernel-equiv harness sets this to the root,
    // shared-linked backend .so that loads against the llvm20-patched bridge
    // without LLVM cl::opt double-registration; a per-example build otherwise
    // picks up a divergent copy in its own target/deps that double-registers.
    // TILERS_CODEGEN_SO points at the prebuilt codegen backend dylib; the legacy
    // ASCEND_RUSTC_CODEGEN_TILE_SO name is still honored as a fallback.
    if let Ok(p) =
        env::var("TILERS_CODEGEN_SO").or_else(|_| env::var("ASCEND_RUSTC_CODEGEN_TILE_SO"))
    {
        let pb = PathBuf::from(&p);
        if pb.is_file() {
            return pb;
        }
    }
    let filename = format!(
        "{}rustc_codegen_tile{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_SUFFIX
    );
    for mut path in dylib_path() {
        path.push(&filename);
        if path.is_file() {
            return path;
        }
    }
    panic!("Could not find {filename} in library path");
}

#[derive(Deserialize, Debug)]
struct RustcOutput {
    reason: String,
    filenames: Option<Vec<String>>,
}

/// Search cargo build output for a codegen artifact with the given extension suffix.
fn find_codegen_output(cargo_stdout: &str, suffix: &str) -> Option<PathBuf> {
    let artifacts =
        cargo_stdout
            .lines()
            .filter_map(|line| match serde_json::from_str::<RustcOutput>(line) {
                Ok(line) => Some(line),
                Err(_) => {
                    println!("{line}");
                    None
                }
            });

    let artifacts = artifacts
        .filter(|line| line.reason == "compiler-artifact")
        .collect::<Vec<_>>();

    for artifact in artifacts {
        let filenames = artifact.filenames.unwrap_or_default();
        if !filenames.is_empty() {
            let path = PathBuf::from(&filenames[0]);
            let dir = path.parent().unwrap();

            for entry in std::fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();

                if path.to_str().unwrap().ends_with(suffix) {
                    return Some(path);
                }
            }
        }
    }

    None
}
