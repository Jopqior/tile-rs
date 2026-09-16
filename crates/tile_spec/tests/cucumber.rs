//! Executable Gherkin harness for tile-rs.
//!
//! Loads every `.feature` under `features/`, registers step definitions that
//! drive the REAL open surface:
//!   * the `CodegenTarget` trait + `TargetRegistry` (from `tile_codegen`),
//!   * the Metal (MSL) emitter registered in `tile_codegen`,
//!   * the tile DSL public signatures (`tile_std/src/tile.rs`, read as text
//!     because that crate is `no_core` and not host-runnable).
//!
//! This file includes no emitter source. The step definitions live in
//! `tile_spec::steps` so other crates can run scenarios against their targets.
//!
//! Each `Scenario` becomes one `#[test]`-equivalent assertion run: the harness
//! reports "<N> scenarios, all green", and any UNDEFINED or failing step fails
//! the test with the scenario name + step text.

use std::path::Path;
use tile_codegen::TargetRegistry;
use tile_spec::steps::{build_runner, run_feature_dir};

/// Discover and run every `.feature` file, asserting all scenarios green.
#[test]
fn all_features_green() {
    let runner = build_runner(TargetRegistry::with_builtin);
    let (features_run, total_scenarios) = run_feature_dir(&runner, Path::new("features"));

    eprintln!(
        "tile-rs GWT: {features_run} feature file(s), {total_scenarios} scenario(s) all green"
    );
    assert!(total_scenarios >= 1);
}
