//! External behaviour of the independent Metal correctness entry.
//!
//! These tests drive the `metal-correctness` binary only. They do not reach
//! into emitter internals. GPU execution is the CI job on macos-15; these
//! tests cover self-check, the delivered case list, and the workflow contract.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn bin() -> Command {
    let path = option_env!("CARGO_BIN_EXE_metal_correctness")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
                "target/{}/metal-correctness",
                if cfg!(debug_assertions) { "debug" } else { "release" }
            ))
        });
    Command::new(path)
}

fn output_text(out: &std::process::Output) -> String {
    let mut s = String::new();
    s.push_str(&String::from_utf8_lossy(&out.stdout));
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

#[test]
fn self_check_rejects_obvious_add_mismatch_and_reports_separately_from_gpu() {
    let out = bin()
        .arg("--self-check")
        .output()
        .expect("run metal-correctness --self-check");
    let text = output_text(&out);
    assert!(
        out.status.success(),
        "self-check must succeed only when injections are rejected\n{text}"
    );
    assert!(
        text.contains("SELF_CHECK=pass"),
        "self-check evidence missing\n{text}"
    );
    assert!(
        text.contains("SELF_CHECK_MISMATCH=rejected"),
        "obvious add mismatch was not rejected\n{text}"
    );
    assert!(
        !text.contains("GPU_VERIFY=pass"),
        "self-check must not report GPU verification as pass\n{text}"
    );
}

#[test]
fn delivered_case_list_names_the_add_case_without_running_gpu() {
    let out = bin()
        .arg("--list")
        .output()
        .expect("run metal-correctness --list");
    let text = output_text(&out);
    assert!(out.status.success(), "--list failed\n{text}");
    assert!(
        text.contains("add_f32_small"),
        "expected delivered case missing from list\n{text}"
    );
    assert!(
        !text.contains("GPU_VERIFY=pass"),
        "--list must not count as GPU verification\n{text}"
    );
}

#[test]
fn self_check_detects_a_missing_required_case() {
    let out = bin()
        .arg("--self-check")
        .output()
        .expect("run metal-correctness --self-check");
    let text = output_text(&out);
    assert!(out.status.success(), "self-check failed\n{text}");
    assert!(
        text.contains("SELF_CHECK_MISSING_CASE=rejected"),
        "missing-case injection was not rejected\n{text}"
    );
}

#[test]
fn unknown_flag_is_not_success() {
    let out = bin()
        .arg("--not-a-real-flag")
        .output()
        .expect("run metal-correctness with unknown flag");
    assert!(
        !out.status.success(),
        "unknown flag must not succeed"
    );
}

#[test]
fn single_case_reuses_current_emitter_convert_mlir_to_msl() {
    let out = bin()
        .args(["--case", "add_f32_small"])
        .output()
        .expect("run metal-correctness --case");
    let text = output_text(&out);
    assert!(
        text.contains("EMIT=pass interface=convert_mlir_to_msl"),
        "entry must call convert_mlir_to_msl\n{text}"
    );
    assert!(
        text.contains("add_f32_small"),
        "case id missing\n{text}"
    );
}

#[test]
fn single_case_mode_is_not_a_complete_list_run() {
    let out = bin()
        .args(["--case", "add_f32_small"])
        .output()
        .expect("run metal-correctness --case");
    let text = output_text(&out);
    assert!(
        text.contains("COMPLETE_LIST=no"),
        "single-case mode must declare it is not the full list\n{text}"
    );
    assert!(
        text.contains("CI_EQUIVALENT=no"),
        "single-case mode must not claim CI equivalence\n{text}"
    );
}

#[test]
fn workflow_runs_the_full_delivered_list_on_free_macos15() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join(".github/workflows/metal-correctness.yml");
    let yaml = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    assert!(
        yaml.contains("pull_request:"),
        "workflow must run on every PR"
    );
    assert!(
        yaml.contains("workflow_dispatch:"),
        "workflow must allow manual trigger"
    );
    assert!(
        yaml.contains("branches: [main]") || yaml.contains("branches: [ main ]"),
        "workflow must run on main updates\n{yaml}"
    );
    assert!(
        yaml.contains("macos-15"),
        "workflow must use the free macos-15 runner"
    );
    assert!(
        !yaml.contains("paths:"),
        "workflow must not path-filter\n{yaml}"
    );
    assert!(
        !yaml.contains("continue-on-error"),
        "workflow must not continue-on-error"
    );
    assert!(
        !yaml.contains("--case"),
        "CI must not use single-case mode as the full list"
    );
    assert!(
        yaml.contains("metal-correctness") || yaml.contains("cargo run"),
        "workflow must invoke the verification entry"
    );
    assert!(
        yaml.to_lowercase().contains("not") && yaml.contains("#31"),
        "workflow must say initial coverage is not #31 complete\n{yaml}"
    );
}
