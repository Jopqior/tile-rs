//! Exposes the unit tests embedded in the case model.
//!
//! The `metal-correctness` bin target disables the test harness (the
//! `#[path]`-included emitter ships its own `#[cfg(test)]` module that needs
//! files this crate does not provide), so `cargo test` would otherwise never
//! compile `cases.rs`'s tests. Including it from an integration test target
//! turns `cfg(test)` on for that module.

#[path = "../src/cases.rs"]
mod cases;
