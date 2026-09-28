pub mod compiler;
pub mod ffi;
pub mod target;
pub mod validate;

use std::path::Path;

use anyhow::Result;

pub use compiler::{CompileConfig, compile_cube_kernel};
pub use target::{AscendTarget, FlagStyle, OutputFormat};
pub use validate::{Severity, ValidationDiagnostic};

/// Errors from the compilation pipeline.
#[derive(Debug)]
pub enum CompileError {
    /// Validation failed with one or more errors.
    Validation(Vec<ValidationDiagnostic>),
    /// CANN SDK / environment setup error.
    Setup(String),
    /// Bisheng compilation failed.
    Compiler(String),
    /// I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Validation(diags) => {
                write!(f, "validation failed:")?;
                for d in diags {
                    write!(f, "\n  {}", d)?;
                }
                Ok(())
            }
            CompileError::Setup(msg) => write!(f, "setup error: {}", msg),
            CompileError::Compiler(msg) => write!(f, "compiler error: {}", msg),
            CompileError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<std::io::Error> for CompileError {
    fn from(e: std::io::Error) -> Self {
        CompileError::Io(e)
    }
}

/// Compile a C++ kernel from source string. Returns the compiled binary bytes.
///
/// If `config.validate` is true, runs validation first and fails on any `Error`-severity finding.
pub fn compile_kernel(source: &str, config: &CompileConfig) -> Result<Vec<u8>, CompileError> {
    // Validate
    if config.validate {
        let diags = validate::validate_kernel(source, config.target);
        let errors: Vec<_> = diags
            .into_iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        if !errors.is_empty() {
            return Err(CompileError::Validation(errors));
        }
    }

    // Write source to temp file
    let tmp_dir = tempfile::tempdir().map_err(|e| CompileError::Io(e))?;
    let input_path = tmp_dir.path().join("kernel.cpp");
    std::fs::write(&input_path, source)?;

    let ext = match config.output_format {
        OutputFormat::Object => "o",
        OutputFormat::SharedLib => "so",
    };
    let output_path = tmp_dir.path().join(format!("kernel.{}", ext));

    // Build and execute
    let settings = tile_kernel_builder_config::Settings::new()
        .map_err(|e| CompileError::Setup(e.to_string()))?;

    let cmd = compiler::build_bisheng_command(&settings, &input_path, &output_path, config)
        .map_err(|e| CompileError::Setup(e.to_string()))?;

    compiler::execute_bisheng(cmd).map_err(|e| CompileError::Compiler(e.to_string()))?;

    // Read result
    let bytes = std::fs::read(&output_path)?;
    Ok(bytes)
}

/// Compile a C++ kernel file to an output file.
pub fn compile_kernel_file(
    input: &Path,
    output: &Path,
    config: &CompileConfig,
) -> Result<(), CompileError> {
    // Validate
    if config.validate {
        let source = std::fs::read_to_string(input)?;
        let diags = validate::validate_kernel(&source, config.target);
        let errors: Vec<_> = diags
            .into_iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        if !errors.is_empty() {
            return Err(CompileError::Validation(errors));
        }
    }

    // Build and execute
    let settings = tile_kernel_builder_config::Settings::new()
        .map_err(|e| CompileError::Setup(e.to_string()))?;

    let cmd = compiler::build_bisheng_command(&settings, input, output, config)
        .map_err(|e| CompileError::Setup(e.to_string()))?;

    compiler::execute_bisheng(cmd).map_err(|e| CompileError::Compiler(e.to_string()))?;

    Ok(())
}

/// Compile a cube kernel C++ file to a shared library (.so) using the
/// AIC+AIV dual compilation pipeline.
pub fn compile_cube_kernel_file(
    input: &Path,
    output_so: &Path,
    kernel_name: &str,
    config: &CompileConfig,
) -> Result<(), CompileError> {
    compiler::compile_cube_kernel(input, output_so, kernel_name, config)
        .map_err(|e| CompileError::Compiler(e.to_string()))
}

/// Run validation passes on C++ kernel source without compiling.
pub fn validate_kernel(source: &str, target: AscendTarget) -> Vec<ValidationDiagnostic> {
    validate::validate_kernel(source, target)
}
