use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::{env, fmt};

pub const DEFAULT_SOC_VERSION: &'static str = "Ascend310P1";

// FIXME: add cpu stubs mode
#[derive(Clone, Debug)]
pub enum RunMode {
    Npu(String),
    Sim(String),
}

impl RunMode {
    pub fn soc_version(&self) -> &str {
        match &self {
            RunMode::Npu(version) => version,
            RunMode::Sim(version) => version,
        }
    }

    /// Map SoC version string to the `bisheng` C++ compiler's
    /// `--cce-aicore-arch` value.  Used by `cpp_builder.rs` (cmake
    /// path).
    pub fn cce_aicore_arch(&self) -> Option<&'static str> {
        match self.soc_version() {
            s if s.starts_with("Ascend310P") => Some("dav-m200"),
            s if s.starts_with("Ascend310B") => Some("dav-m200i"),
            "Ascend310" => Some("dav-c100"),
            // dav-c220s is used in newer CANN (>= 8.1); older CANN (e.g. 8.5.0 from 910c)
            // only supports dav-c220. Use the shorter form which works on both versions.
            s if s.starts_with("Ascend910B") => Some("dav-c220"),
            s if s.starts_with("Ascend910Pro") => Some("dav-c100"),
            "Ascend910" | "Ascend910A" => Some("dav-c100"),
            // Ascend910_9362..9392 are Ascend910B chips (dav-c220), NOT original Ascend910 (dav-c100).
            // aclrtGetSocName() returns "Ascend910_9392" on 910c hardware which is dav-c220.
            s if s.starts_with("Ascend910_") => Some("dav-c220"),
            _ => None,
        }
    }

    /// Whether this target requires the C++ codegen path instead of
    /// `bishengir-compile`.
    ///
    /// On Ascend310P, `bishengir-compile` doesn't produce functional kernel
    /// binaries: cross-object calls fail (the AICore has no call stack), and
    /// the tool doesn't generate proper pipe synchronization. The validated
    /// approach is to generate C++ code with AscendC DMA wrappers and
    /// `pipe_barrier(PIPE_ALL)`, compiled by `bisheng` as a single TU.
    pub fn needs_cpp_codegen(&self) -> bool {
        // All targets use C++ codegen: the generated code uses AscendC APIs
        // (DataCopy, Muls, pipe_barrier) compiled by bisheng as a single TU.
        // The alternative bishengir-compile MLIR path doesn't handle DMA
        // synchronization or pipe barriers properly.
        true
    }

    /// Map SoC version string to `bishengir-compile --target` value.
    ///
    /// `bishengir-compile` only supports Ascend910 series targets:
    ///   Ascend910B1..B4, Ascend910_9362..9392
    /// For all other chips (Ascend310, Ascend310P, etc.) return `None`
    /// — `bishengir-compile` infers the target automatically.
    pub fn target_arch(&self) -> Option<&'static str> {
        match self.soc_version() {
            "Ascend910B1" => Some("Ascend910B1"),
            "Ascend910B2" => Some("Ascend910B2"),
            "Ascend910B3" => Some("Ascend910B3"),
            "Ascend910B4" => Some("Ascend910B4"),
            s if s.starts_with("Ascend910_") => {
                // Pass through Ascend910_XXXX values directly
                // Leak a static string since these are compile-time constants
                // from the environment variable.
                None // For safety, don't pass unknown Ascend910_ variants
            }
            _ => None,
        }
    }
}

impl fmt::Display for RunMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            RunMode::Npu(_) => write!(f, "npu"),
            RunMode::Sim(_) => write!(f, "sim"),
        }
    }
}

// FIXME: all getters have to return pre-computed paths.
#[derive(Debug)]
pub struct Settings {
    cann: PathBuf,
    run_mode: RunMode,
    target_prefix: &'static str,
    // FIXME: I don't know how to name this correctly.
    // Path example: /usr/local/Ascend/ascend-toolkit/8.1.RC1/
    ascend_toolkit: PathBuf,
    cann_include_paths: Vec<PathBuf>,
}

impl Settings {
    pub fn new() -> Result<Settings> {
        let ascend_cann_path = Self::init_cann_path()?;
        let run_mode = Self::init_run_mode()?;
        let target_prefix = Self::init_target_prefix(&ascend_cann_path)?;
        let ascend_toolkit = Self::init_ascend_tookkit_path(&ascend_cann_path)?;
        let cann_include_paths = Self::init_cann_include_paths(&ascend_cann_path, target_prefix);

        Ok(Settings {
            cann: ascend_cann_path,
            run_mode,
            target_prefix,
            ascend_toolkit,
            cann_include_paths,
        })
    }

    #[inline]
    pub fn cann(&self) -> &Path {
        &self.cann
    }

    #[inline]
    pub fn run_mode(&self) -> &RunMode {
        &self.run_mode
    }

    #[inline]
    pub fn target_prefix(&self) -> &'static str {
        self.target_prefix
    }

    #[inline]
    pub fn cann_ascend_lib(&self) -> PathBuf {
        self.cann.join(&self.target_prefix).join("lib64")
    }

    #[inline]
    pub fn cann_devlib(&self) -> PathBuf {
        self.cann.join(&self.target_prefix).join("devlib")
    }

    #[inline]
    pub fn cann_ascendnpu_ir(&self) -> PathBuf {
        PathBuf::from(format!(
            "{}/{}/bin/bishengir-compile",
            self.cann.display(),
            self.target_prefix
        ))
    }

    /// Path to the `bisheng` C++ compiler (clang-based CCE frontend).
    ///
    /// Used by the C++ codegen path for targets where `bishengir-compile`
    /// doesn't produce functional binaries (e.g. Ascend310P).
    #[inline]
    pub fn cann_bisheng(&self) -> PathBuf {
        self.cann.join("tools/ccec_compiler/bin/bisheng")
    }

    /// Path to the CANN version header, needed for bisheng compilation.
    /// This is architecture-independent (no target prefix).
    #[inline]
    pub fn cann_version_header(&self) -> PathBuf {
        self.cann.join("include/version/cann_version.h")
    }

    /// Include paths needed for bisheng C++ kernel compilation (tikcpp framework).
    pub fn cann_bisheng_include_paths(&self) -> Vec<PathBuf> {
        vec![
            self.cann.join("tools/tikcpp/tikcfw"),
            self.cann.join("tools/tikcpp/tikcfw/interface"),
            self.cann.join("tools/tikcpp/tikcfw/impl"),
            self.cann.join("tools/tikcpp/tikcfw/impl/aicore"),
            self.cann.join("tools/tikcpp/tikcfw/impl/aicore/basic_api"),
        ]
    }

    /// Extra include paths needed when bisheng compiles ptoas-generated C++.
    ///
    /// ptoas emits `#include "pto/pto-inst.hpp"` which lives under
    /// `{cann}/{target_prefix}/include/`.  This path is NOT in the standard
    /// bisheng include list because it is not needed for the plain C++ path.
    pub fn cann_pto_include_paths(&self) -> Vec<PathBuf> {
        vec![self.cann.join(self.target_prefix).join("include")]
    }

    #[inline]
    pub fn cann_sim(&self) -> Option<PathBuf> {
        match &self.run_mode {
            RunMode::Sim(platform) => {
                Some(format!("{}/tools/simulator/{}/lib", self.cann.display(), platform).into())
            }
            RunMode::Npu(_) => None,
        }
    }

    #[inline]
    pub fn cann_ops_lib(&self) -> PathBuf {
        self.ascend_toolkit.join(&self.target_prefix).join("lib")
    }

    #[inline]
    pub fn cann_ops_include(&self) -> PathBuf {
        PathBuf::from(format!(
            "{}/{}/include/acl/ops/",
            self.ascend_toolkit.display(),
            &self.target_prefix
        ))
    }

    pub fn cann_include_paths(&self) -> &[PathBuf] {
        &self.cann_include_paths
    }

    // Determine the CANN package path
    fn init_cann_path() -> Result<PathBuf> {
        // Determine the CANN package path
        let ascend_cann_path: PathBuf  = env::var("ACLRS_CANN_PATH").ok().and_then(|path|{
            let path = PathBuf::from(path);
            path.exists().then_some(path)
        }).or_else(|| {
            env::var("HOME").ok().and_then(|home| {
                let default_user_path = PathBuf::from(format!("{}/Ascend/ascend-toolkit/latest", home));
                default_user_path.exists().then_some(default_user_path)
            }).or_else(|| {
                let default_root_path = PathBuf::from(format!("/usr/local/Ascend/ascend-toolkit/latest"));
                default_root_path.exists().then_some(default_root_path)
            })
        }).ok_or(anyhow!("Could not determine ASCEND_HOME_PATH. Please set it or ensure a standard installation path."))?;
        Ok(ascend_cann_path)
    }

    fn init_target_prefix(ascend_cann_path: impl AsRef<Path>) -> Result<&'static str> {
        const X86: &'static str = "x86_64-linux";
        const AARCH: &'static str = "aarch64-linux";

        let ascend_cann_path = ascend_cann_path.as_ref();
        if ascend_cann_path.join(X86).exists() {
            return Ok(X86);
        } else if ascend_cann_path.join(AARCH).exists() {
            return Ok(AARCH);
        } else {
            panic!("Unsupported target architecture")
        }
    }

    // Determine the execution mode
    fn init_run_mode() -> Result<RunMode> {
        let soc_version = env::var("ACLRS_SOC_VERSION").unwrap_or(DEFAULT_SOC_VERSION.to_string());
        let run_mode = match env::var("ACLRS_RUN_MODE").ok() {
            Some(run_mode) => match run_mode.as_str() {
                "npu" => RunMode::Npu(soc_version),
                "sim" => {
                    // FIXME: check that soc_version is valid
                    RunMode::Sim(soc_version)
                }
                _ => panic!("Unexpected run mode: {}", run_mode),
            },
            None => RunMode::Npu(soc_version),
        };
        Ok(run_mode)
    }

    fn init_ascend_tookkit_path(ascend_cann_path: impl AsRef<Path>) -> Result<PathBuf> {
        let cann_path = ascend_cann_path.as_ref();
        let config = cann_path.join("version.cfg");
        if !config.is_file() {
            // No version.cfg — fall back to the cann path itself.
            return Ok(cann_path.to_path_buf());
        }
        let config = std::fs::read_to_string(config)?;
        let re = regex::Regex::new(r"\[(.*):(.*)\]")?;
        let caps = re
            .captures(&config)
            .ok_or(anyhow!("couldn't read CANN config"))?;
        let matches = caps.get(2).ok_or(anyhow!("couldn't read CANN config"))?;

        let version = matches.as_str();
        let mut cann = cann_path.components();
        cann.next_back();
        let versioned = cann.as_path().join(version);
        // If the versioned directory doesn't exist (e.g. partial install),
        // fall back to the cann path itself.
        if versioned.exists() {
            Ok(versioned)
        } else {
            Ok(cann_path.to_path_buf())
        }
    }

    fn init_cann_include_paths(
        ascend_cann_path: impl AsRef<Path>,
        target_prefix: &'static str,
    ) -> Vec<PathBuf> {
        vec![
            PathBuf::from(format!(
                "{}/tools/tikcpp/tikcfw",
                ascend_cann_path.as_ref().display()
            )),
            PathBuf::from(format!(
                "{}/tools/tikcpp/tikcfw/interface",
                ascend_cann_path.as_ref().display()
            )),
            PathBuf::from(format!(
                "{}/tools/tikcpp/tikcfw/impl",
                ascend_cann_path.as_ref().display()
            )),
            PathBuf::from(format!(
                "{}/{}/include",
                ascend_cann_path.as_ref().display(),
                target_prefix
            )),
            // CANN >= 8.5: rt.h moved to pkg_inc/runtime/runtime/rt.h;
            // add pkg_inc/runtime as a search root so the include
            // "#include <runtime/runtime/rt.h>" resolves correctly.
            PathBuf::from(format!(
                "{}/{}/pkg_inc/runtime",
                ascend_cann_path.as_ref().display(),
                target_prefix
            )),
            PathBuf::from(format!(
                "{}/{}/include/experiment/msprof",
                ascend_cann_path.as_ref().display(),
                target_prefix
            )),
            PathBuf::from(format!(
                "{}/{}/include/experiment/runtime/runtime",
                ascend_cann_path.as_ref().display(),
                target_prefix
            )),
        ]
    }
}

// FIXME: how to propagate RPATH(or setup LD_LIBRARY_PATH) without
// additional API call from user site?
pub fn add_ascend_link_args() -> Result<()> {
    let settings = Settings::new()?;
    if let Some(sim) = settings.cann_sim() {
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", sim.display());
    }
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath={}",
        settings.cann_ascend_lib().display()
    );
    // NOTE: Do NOT add cann_devlib() to rpath. It contains a stub
    // libascend_hal.so that shadows the real driver HAL from
    // /usr/local/Ascend/driver/lib64/driver/. The real HAL is found
    // via LD_LIBRARY_PATH set by setenv.bash.

    Ok(())
}
