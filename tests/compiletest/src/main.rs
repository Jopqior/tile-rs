use std::env;
use std::fmt;
use std::path::{Path, PathBuf};

const TARGET: &str = "davinci-huawei-none";

#[derive(Debug)]
enum FindLibError {
    Missing,
    Duplicate,
}

impl fmt::Display for FindLibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FindLibError::Missing => write!(f, "missing"),
            FindLibError::Duplicate => write!(f, "duplicate"),
        }
    }
}

#[derive(Debug)]
enum CompiletestError {
    CodegenNotFound(PathBuf),
    CantBuildDeps(PathBuf),
    FindLibError(PathBuf, FindLibError),
}

impl std::error::Error for CompiletestError {}

impl fmt::Display for CompiletestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompiletestError::CodegenNotFound(path) => {
                write!(f, "Codegen backend not found: {}", path.display())
            }
            CompiletestError::CantBuildDeps(path) => {
                write!(f, "Failed to build test dependencies: {}", path.display())
            }
            CompiletestError::FindLibError(lib, err) => {
                write!(f, "Can't find lib {}. Reason: {}.", lib.display(), err)
            }
        }
    }
}

#[derive(Copy, Clone)]
enum DepKind {
    AscendLib,
    ProcMacro,
}

impl DepKind {
    fn prefix_and_extension(self) -> (&'static str, &'static str) {
        match self {
            Self::AscendLib => ("lib", "rlib"),
            Self::ProcMacro => (env::consts::DLL_PREFIX, env::consts::DLL_EXTENSION),
        }
    }

    fn target_dir_suffix(self, target: &str) -> String {
        match self {
            Self::AscendLib => format!("{target}/debug/deps"),
            Self::ProcMacro => "debug/deps".into(),
        }
    }
}

fn main() -> Result<(), CompiletestError> {
    let runner = TestRunner::new()?;
    runner.run_mode("ui")?;
    Ok(())
}

struct TestRunner {
    /// `tile-rs` project workspace.
    workspace_root: PathBuf,
    /// Current crate with `compiletest`-based testing infrastructure.
    ///
    /// Location: ${workspace_root}/tests/.
    tests_dir: PathBuf,
    /// `compiletest`'s output.
    ///
    /// Location: ${workspace_root}/target/compiletest_results.
    compiletest_build_dir: PathBuf,
    /// Contains tests dependencies including: 'tile_std' and 'tile_std_macros'.
    ///
    /// Location: ${workspace_root}/target/compiletest_deps.
    deps_target_dir: PathBuf,
    /// Compiled backend.
    codegen_backend_path: PathBuf,
}

impl TestRunner {
    fn new() -> Result<TestRunner, CompiletestError> {
        let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = tests_dir.parent().unwrap().parent().unwrap().to_path_buf();
        let original_target_dir = workspace_root.join("target");
        let deps_target_dir = original_target_dir.join("compiletest_deps");
        let compiletest_build_dir = original_target_dir.join("compiletest_results");
        let codegen_backend_path = find_rustc_codegen_tile();

        // Custom target spec: tell rustc where to find davinci-huawei-none.json
        //
        // SAFETY: single-threaded startup, before any other threads exist
        // (edition 2024 made env::set_var unsafe; the call itself is unchanged).
        unsafe {
            std::env::set_var("RUST_TARGET_PATH", &workspace_root);
        }

        // HACK(eddyb) force `compiletest` to pass `ui/...` relative paths to `rustc`,
        // which should always end up being the same regardless of the path that the
        // Rust CUDA repo is checked out at (among other things, this avoids hardcoded
        // `compiletest` limits being hit by e.g. users with slightly longer paths).
        std::env::set_current_dir(tests_dir).unwrap();
        let tests_dir = PathBuf::from("");

        Ok(TestRunner {
            tests_dir,
            workspace_root,
            compiletest_build_dir,
            deps_target_dir,
            codegen_backend_path,
        })
    }

    fn run_mode(&self, mode: &'static str) -> Result<(), CompiletestError> {
        let deps = self.build_deps(TARGET)?;

        let deps_dir = &[
            &self
                .deps_target_dir
                .join(DepKind::AscendLib.target_dir_suffix(TARGET)),
            &self
                .deps_target_dir
                .join(DepKind::ProcMacro.target_dir_suffix(TARGET)),
        ];

        println!("{}", deps.tile_std_macros.display());
        let mut target_rustcflags = vec![
            deps_dir
                .iter()
                .map(|dir| format!("-L dependency={}", dir.display()))
                .fold(String::new(), |a, b| b + " " + &a),
            format!(
                "--extern tile_std_macros={}",
                deps.tile_std_macros.display()
            ),
            format!("--extern tile_std={}", deps.tile_std.display()),
            "--crate-type cdylib".into(),
            "-Zui-testing".into(),
            "--edition=2024".into(),
        ];
        target_rustcflags.append(&mut self.get_rustcflags_for_tests());
        let target_rustcflags = target_rustcflags.join(" ");

        let config = compiletest_rs::Config {
            target_rustcflags: Some(target_rustcflags),
            mode: mode.parse().expect("Invalid mode"),
            target: TARGET.to_string(),
            src_base: self.tests_dir.join(mode),
            build_base: self.compiletest_build_dir.clone(),
            bless: false,
            ..compiletest_rs::Config::default()
        };

        compiletest_rs::run_tests(&config);

        Ok(())
    }

    fn get_rustcflags_for_tests(&self) -> Vec<String> {
        [
            format!("-Zcodegen-backend={}", self.codegen_backend_path.display()),
            // Ensure the codegen backend is emitted in `.d` files to force Cargo
            // to rebuild crates compiled with it when it changes (this used to be
            // the default until https://github.com/rust-lang/rust/pull/93969).
            "-Zbinary-dep-depinfo".into(),
            "-Zcrate-attr=feature(register_tool)".into(),
            "-Zcrate-attr=register_tool(tile)".into(),
            // "-Zcrate-attr=no_std".into(),
            "-Zsaturating_float_casts=false".into(),
            "-Cpanic=abort".into(), // `panic_immediate_abort` requires that
        ]
        .to_vec()
    }

    fn build_deps(&self, target: &str) -> Result<TestDeps, CompiletestError> {
        let rustflags = self.get_rustcflags_for_tests();
        let cargo_encoded_rustflags = rustflags.join("\x1f");

        std::process::Command::new("cargo")
            .args([
                "build",
                "-p",
                "tile_std",
                // "--release",
                "--lib",
                "-Zbuild-std-features=panic_immediate_abort",
                // "-Zbuild-std=core",
                // "-Zbuild-std-features=compiler-builtins-mem",
                &*format!("--target={target}"),
            ])
            .arg("--target-dir")
            .arg(&self.deps_target_dir)
            .env("CARGO_ENCODED_RUSTFLAGS", cargo_encoded_rustflags)
            .env("RUST_TARGET_PATH", &self.workspace_root)
            .stderr(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::inherit())
            .status()
            .expect("failed to execute cargo build");

        let tile_std = find_lib(
            &self.deps_target_dir,
            "tile_std",
            DepKind::AscendLib,
            target,
        )?;
        let tile_std_macros = find_lib(
            &self.deps_target_dir,
            "tile_std_macros",
            DepKind::ProcMacro,
            target,
        )?;

        Ok(TestDeps {
            tile_std,
            tile_std_macros,
        })
    }
}

/// Attempt find the rlib that matches `base`, if multiple rlibs are found then
/// a clean build is required and `Err(FindLibError::Duplicate)` is returned.
fn find_lib(
    deps_target_dir: &Path,
    base: impl AsRef<Path>,
    dep_kind: DepKind,
    target: &str,
) -> Result<PathBuf, CompiletestError> {
    let base = base.as_ref();
    let (expected_prefix, expected_extension) = dep_kind.prefix_and_extension();
    let expected_name = format!("{}{}", expected_prefix, base.display());

    let dir = deps_target_dir.join(dep_kind.target_dir_suffix(target));

    let matching_paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            let name = {
                let name = path.file_stem();
                if name.is_none() {
                    return false;
                }
                name.unwrap()
            };

            let name_matches = name.to_str().unwrap().starts_with(&expected_name)
                && name.len() == expected_name.len() + 17   // we expect our name, '-', and then 16 hexadecimal digits
                && ends_with_dash_hash(name.to_str().unwrap());
            let extension_matches = path
                .extension()
                .is_some_and(|ext| ext == expected_extension);

            name_matches && extension_matches
        })
        .collect();

    match matching_paths.len() {
        0 => Err(CompiletestError::FindLibError(
            base.to_path_buf(),
            FindLibError::Missing,
        )),
        1 => Ok(matching_paths.into_iter().next().unwrap()),
        _ => Err(CompiletestError::FindLibError(
            base.to_path_buf(),
            FindLibError::Duplicate,
        )),
    }
}

/// Paths to dependencies needed for tests compilation.
struct TestDeps {
    tile_std: PathBuf,
    tile_std_macros: PathBuf,
}

/// Returns whether this string ends with a dash ('-'), followed by 16 lowercase hexadecimal characters
fn ends_with_dash_hash(s: &str) -> bool {
    let n = s.len();
    if n < 17 {
        return false;
    }
    let mut bytes = s.bytes().skip(n - 17);
    if bytes.next() != Some(b'-') {
        return false;
    }

    bytes.all(|b| b.is_ascii_hexdigit())
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
    match std::env::var_os(dylib_path_envvar()) {
        Some(var) => std::env::split_paths(&var).collect(),
        None => Vec::new(),
    }
}

fn find_rustc_codegen_tile() -> PathBuf {
    let filename = format!(
        "{}rustc_codegen_tile{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    for mut path in dylib_path() {
        path.push(&filename);
        if path.is_file() {
            return path;
        }
    }
    panic!("Could not find {filename} in library path");
}
