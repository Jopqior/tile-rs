//! # tile_codegen — the open tile-rs codegen skeleton
//!
//! This crate is the **open core** of tile-rs: the generic, vendor-neutral
//! code-generation infrastructure, with **zero** LLVM / MLIR / melior / rustc
//! dependencies. Everything target-specific plugs in through one trait
//! ([`CodegenTarget`]) and one registry ([`TargetRegistry`]).
//!
//! ## What lives where
//!
//! * **this crate** — the trait, the registry, the `DebugTarget` reference
//!   target and, under the `emitters` feature, the Apple Metal (MSL) emitter
//!   with the std-only MLIR parser it reads.
//! * **every other backend** — registered by the host that has it, through
//!   [`TargetRegistry::register`]. The prebuilt codegen backend registers all of
//!   its targets this way, so they are selectable by the same name.
//!
//! ## Build matrix
//!
//! | features    | builds where     | contains                                |
//! |-------------|------------------|-----------------------------------------|
//! | *(default)* | anywhere         | trait + registry + `DebugTarget`        |
//! | `emitters`  | anywhere         | + the Metal (`msl`) emitter and parser  |
//!
//! The default build is what keeps the skeleton verifiable standalone — see the
//! tests at the bottom of this file.

pub mod registry;
pub mod target;
pub mod targets;

pub use registry::TargetRegistry;
pub use target::{CodegenTarget, EmitOpts, EmitOut, HardwareParams, TargetMeta};
pub use targets::{DebugTarget, EmitterTarget};

// Under `emitters`: the shared std-only parser the real emitters import as
// `crate::mlir_parse`, plus the emitter source modules. Compiled on an LLVM-20
// box (the emitters live in the `rustc_codegen_tile` tree).
#[cfg(feature = "emitters")]
#[path = "../../rustc_codegen_tile/src/mlir_parse.rs"]
#[allow(dead_code)]
pub(crate) mod mlir_parse;

// The open Metal arm. Declared at the crate root because its sources refer to
// themselves as `crate::mlir_to_msl`.
#[cfg(feature = "emitters")]
#[path = "../../rustc_codegen_tile/src/mlir_to_msl.rs"]
#[allow(dead_code)]
pub(crate) mod mlir_to_msl;

#[cfg(feature = "emitters")]
pub(crate) mod emitters;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_populated_and_selectable() {
        let r = TargetRegistry::with_builtin();
        assert!(
            !r.is_empty(),
            "registry should have at least the debug target"
        );
        assert!(
            r.select("debug").is_some(),
            "debug target must be registered"
        );
        assert!(r.select("does-not-exist").is_none());
        assert!(r.names().contains(&"debug"));
    }

    #[test]
    fn debug_target_emits_source() {
        let r = TargetRegistry::with_builtin();
        let t = r.select("debug").expect("debug target");
        let out = t
            .emit("func.func @kernel() { return }", &EmitOpts::default())
            .expect("emit ok");
        assert!(out.source.contains("func.func"));
        assert!(out.source.contains("tile-rs debug target"));
        assert_eq!(out.ext, "mlir.txt");
    }

    #[test]
    fn empty_mlir_is_an_error() {
        let r = TargetRegistry::with_builtin();
        let t = r.select("debug").unwrap();
        assert!(t.emit("   \n  ", &EmitOpts::default()).is_err());
    }

    #[test]
    fn register_adds_a_custom_target() {
        struct Noop;
        impl CodegenTarget for Noop {
            fn name(&self) -> &'static str {
                "noop"
            }
            fn emit(&self, _m: &str, _o: &EmitOpts) -> Result<EmitOut, String> {
                Ok(EmitOut::default())
            }
        }
        let mut r = TargetRegistry::new();
        assert!(r.is_empty());
        r.register(Box::new(Noop));
        assert_eq!(r.len(), 1);
        assert!(r.select("noop").is_some());
    }

    #[test]
    fn hardware_params_carry_ub_size() {
        // The one asymmetry (AscendC's ub_size) rides on EmitOpts uniformly.
        let opts = EmitOpts {
            hw: HardwareParams {
                ub_size: 192 * 1024,
                ..HardwareParams::default()
            },
        };
        assert_eq!(opts.hw.ub_size, 192 * 1024);
    }

    #[test]
    fn emitter_target_adapter_wraps_a_plain_fn() {
        // This is exactly how `rustc_codegen_tile` registers its existing
        // `convert_mlir_to_*` emitters — a plain
        // `fn(&str) -> Result<String, String>` lifted into the trait, keyed by
        // its TILERS_CODEGEN_PATH name.
        fn fake_convert(mlir: &str) -> Result<String, String> {
            if mlir.is_empty() {
                return Err("empty".into());
            }
            Ok(format!("// CUDA\n{mlir}"))
        }
        let mut r = TargetRegistry::new();
        r.register(Box::new(targets::EmitterTarget::new(
            "cuda",
            "cu",
            fake_convert,
        )));
        let t = r.select("cuda").expect("cuda target registered");
        let out = t.emit("module {}", &EmitOpts::default()).unwrap();
        assert_eq!(out.ext, "cu");
        assert!(out.source.starts_with("// CUDA"));
        assert!(t.emit("", &EmitOpts::default()).is_err());
    }
}
