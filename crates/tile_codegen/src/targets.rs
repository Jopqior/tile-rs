//! Built-in target registration.
//!
//! * The open skeleton always ships a self-contained [`DebugTarget`] so the
//!   trait + registry + emit path is exercisable with zero external deps.
//! * The Metal emitter (`msl`) is wired under the `emitters` feature. See
//!   `crate::emitters::register`.
//! * Every other backend, including the AscendC and PTO targets, is registered
//!   by a host crate with the same `register(Box::new(..))` call, through
//!   [`EmitterTarget`] or its own [`CodegenTarget`] impl. Nothing in this crate
//!   names them.

use crate::registry::TargetRegistry;
use crate::target::{CodegenTarget, EmitOpts, EmitOut, TargetMeta};

/// Register every target compiled into this build.
pub fn register_builtin(r: &mut TargetRegistry) {
    r.register(Box::new(DebugTarget));

    #[cfg(feature = "emitters")]
    crate::emitters::register(r);

}

/// Self-contained reference target. Proves the trait/registry/emit path end to
/// end with no LLVM, no parser, no toolchain — this is what makes the open
/// skeleton verifiable on any machine (including this macOS build). Echoes the
/// MLIR back inside a comment banner.
pub struct DebugTarget;

impl CodegenTarget for DebugTarget {
    fn name(&self) -> &'static str {
        "debug"
    }

    fn emit(&self, mlir_text: &str, _opts: &EmitOpts) -> Result<EmitOut, String> {
        if mlir_text.trim().is_empty() {
            return Err("empty MLIR module".to_string());
        }
        let lines = mlir_text.lines().count();
        let funcs = mlir_text.matches("func.func").count();
        Ok(EmitOut {
            source: format!(
                "// tile-rs debug target\n// {lines} MLIR lines, {funcs} func.func op(s)\n{mlir_text}\n"
            ),
            ext: "mlir.txt",
            meta: TargetMeta::default(),
        })
    }
}

/// Adapter that turns a plain `(mlir: &str) -> Result<String, String>` emitter
/// (the 14 uniform backends) into a [`CodegenTarget`]. Std-only and ALWAYS
/// available — host crates (e.g. `rustc_codegen_tile`) register their own
/// `convert_mlir_to_*` functions through it without enabling the `emitters`
/// feature, so the emitter source can stay in exactly one place.
pub struct EmitterTarget {
    name: &'static str,
    ext: &'static str,
    f: fn(&str) -> Result<String, String>,
}

impl EmitterTarget {
    pub fn new(
        name: &'static str,
        ext: &'static str,
        f: fn(&str) -> Result<String, String>,
    ) -> Self {
        Self { name, ext, f }
    }
}

impl CodegenTarget for EmitterTarget {
    fn name(&self) -> &'static str {
        self.name
    }
    fn emit(&self, mlir_text: &str, _opts: &EmitOpts) -> Result<EmitOut, String> {
        let source = (self.f)(mlir_text)?;
        Ok(EmitOut {
            source,
            ext: self.ext,
            meta: TargetMeta::default(),
        })
    }
}
