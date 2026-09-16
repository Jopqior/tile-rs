//! MLIR->source emitters compiled into this crate.
//!
//! GATE: `feature = "emitters"`. The emitter here is the Apple Metal (MSL) arm:
//! `crate::mlir_to_msl` plus the target-neutral `crate::mlir_parse`, both
//! declared at the crate root in `lib.rs` because the emitter sources name each
//! other by crate-root paths.
//!
//! Other backends are not compiled into this crate. A host crate registers them
//! on the same [`TargetRegistry`] through [`EmitterTarget`], the way the codegen
//! backend does. Every emitter exposes
//! `convert_mlir_to_<t>(mlir: &str) -> Result<String, String>`.

pub use crate::mlir_to_msl::convert_mlir_to_msl;

use crate::registry::TargetRegistry;
use crate::targets::EmitterTarget;

/// Register the emitters compiled into this crate.
pub(crate) fn register(r: &mut TargetRegistry) {
    r.register(Box::new(EmitterTarget::new(
        "msl",
        "metal",
        convert_mlir_to_msl,
    )));
}
