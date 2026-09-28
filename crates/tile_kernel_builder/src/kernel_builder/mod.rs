use std::fmt;
use std::path::{Path, PathBuf};

mod cpp_builder;
mod rs_builder;

#[derive(Debug, Clone, Copy)]
enum KernelSource {
    Rust,
    CPP,
}

#[derive(Debug)]
pub enum BuilderError {
    PathDoesntExist(PathBuf),
    BuildFailed,
}

impl std::error::Error for BuilderError {}

impl fmt::Display for BuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuilderError::PathDoesntExist(path) => {
                write!(f, "Path {} does not exist", path.display())
            }
            BuilderError::BuildFailed => f.write_str("Build failed"),
        }
    }
}

// FIXME: support more build options
pub struct KernelBuilder {
    input_kernel: PathBuf,
    /// Whether a kernel is written in C++ or Rust.
    kernel_source: KernelSource,
    /// A path to copy the final file to.
    pub output_dst: Option<PathBuf>,
    /// Optional override for `TILERS_CODEGEN_PATH` on the cargo subprocess.
    /// When `Some("pto")`, the kernel crate compiles via mlir_to_pto. When
    /// `None`, the subprocess inherits whatever the parent has (or defaults
    /// to `cpp`). Used to mix PTO and cpp kernels in one workspace.
    pub codegen_path: Option<String>,
}

impl KernelBuilder {
    pub fn new(input_kernel: impl AsRef<Path>) -> KernelBuilder {
        let input_kernel = input_kernel.as_ref().to_owned();
        // FIXME: smarter way to detect kernel source.
        let kernel_source = match input_kernel.is_dir() {
            true => KernelSource::Rust,
            false => KernelSource::CPP,
        };
        Self {
            input_kernel,
            kernel_source,
            output_dst: None,
            codegen_path: None,
        }
    }

    pub fn copy_to(mut self, path: impl AsRef<Path>) -> Self {
        self.output_dst = Some(path.as_ref().to_path_buf());
        self
    }

    /// Force a specific `TILERS_CODEGEN_PATH` for this kernel's compilation,
    /// independent of the parent process environment. E.g. `codegen_path("pto")`
    /// builds the kernel via the PTO path even when the host app is built
    /// with `TILERS_CODEGEN_PATH=cpp`.
    pub fn codegen_path(mut self, path: impl Into<String>) -> Self {
        self.codegen_path = Some(path.into());
        self
    }

    /// Builds a kernel from source.
    pub fn build(self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let output = match self.kernel_source {
            KernelSource::CPP => cpp_builder::build(self.input_kernel),
            KernelSource::Rust => {
                rs_builder::build_with_env(self.input_kernel, self.codegen_path.as_deref())
            }
        }?;

        println!("cargo:rerun-if-changed={}", output.display());

        if let Some(copy_path) = self.output_dst {
            std::fs::copy(output, &copy_path)?;
            Ok(copy_path)
        } else {
            Ok(output)
        }
    }
}
