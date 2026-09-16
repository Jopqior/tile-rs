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
fn delivered_list_covers_add_sub_mul_exp_and_combination() {
    let out = bin()
        .arg("--list")
        .output()
        .expect("run metal-correctness --list");
    let text = output_text(&out);
    for id in [
        "add_f32_small",
        "sub_f32_small",
        "mul_f32_small",
        "exp_f32_small",
        "add_mul_f32_small",
    ] {
        assert!(text.contains(id), "missing {id} in delivered list\n{text}");
    }
}

#[test]
fn delivered_list_covers_simd_threadgroup_and_tail_boundaries() {
    let out = bin()
        .arg("--list")
        .output()
        .expect("run metal-correctness --list");
    let text = output_text(&out);
    for id in [
        "add_f32_single",
        "add_f32_vec_minus_1",
        "add_f32_vec_plus_1",
        "add_f32_threadgroup_minus_1",
        "add_f32_threadgroup_plus_1",
        "add_f32_multi_group_tail",
        "sub_f32_multi_group_tail",
        "mul_f32_multi_group_tail",
        "exp_f32_single",
        "exp_f32_multi_group_tail",
        "add_mul_f32_multi_group_tail",
    ] {
        assert!(text.contains(id), "missing boundary case {id}\n{text}");
    }
}

#[test]
fn self_check_covers_every_path_and_the_combination_order() {
    let out = bin()
        .arg("--self-check")
        .output()
        .expect("run metal-correctness --self-check");
    let text = output_text(&out);
    assert!(out.status.success(), "self-check failed\n{text}");
    for key in [
        "SELF_CHECK_HAND_ADD=pass",
        "SELF_CHECK_HAND_SUB=pass",
        "SELF_CHECK_HAND_MUL=pass",
        "SELF_CHECK_HAND_EXP=pass",
        "SELF_CHECK_HAND_ADD_MUL=pass",
        "SELF_CHECK_COMBO_ORDER=rejected",
        "SELF_CHECK_TAIL_MISS=rejected",
        "SELF_CHECK_SKIPPED_WRITE=rejected",
        "SELF_CHECK_DROPPED_CASE=rejected",
        "SELF_CHECK_NO_DEVICE_SIZE=rejected",
    ] {
        assert!(text.contains(key), "self-check evidence {key} missing\n{text}");
    }
}

/// Local diagnostic, not GPU evidence: the two-step order also has to be
/// verified numerically by the GPU combination cases on the macos-15 runner.
/// This only catches a locally obvious wrong-association emission off-device.
#[test]
fn combination_case_emits_the_declared_two_step_order_local_diagnostic() {
    let out = bin()
        .args(["--case", "add_mul_f32_small"])
        .env("METAL_CORRECTNESS_DUMP_MSL", "1")
        .output()
        .expect("run metal-correctness --case add_mul_f32_small");
    let text = output_text(&out);
    assert!(
        text.contains("((p0[gid] + p1[gid]) * p2[gid])"),
        "combination MSL does not compute (a+b)*c in order\n{text}"
    );
    assert!(
        !text.contains("GPU_VERIFY=pass"),
        "no GPU run is expected in this off-device check\n{text}"
    );
}

#[test]
fn device_relative_case_is_unverified_not_pass_off_the_full_list() {
    let out = bin()
        .args(["--case", "add_f32_vec"])
        .output()
        .expect("run metal-correctness --case add_f32_vec");
    let text = output_text(&out);
    // Off the full list with no PyTorch reference (and possibly no device), the
    // case must be unverified. It must never claim a GPU pass or exit 0.
    assert_eq!(
        out.status.code(),
        Some(2),
        "a single device-relative case without a verified reference must exit unverified\n{text}"
    );
    assert!(
        text.contains("CASE_UNAVAILABLE")
            || text.contains("GPU_UNVERIFIED")
            || text.contains("REFERENCE_UNVERIFIED"),
        "device-relative case failure reason missing\n{text}"
    );
    assert!(
        !text.contains("GPU_VERIFY=pass"),
        "device-relative case must not claim GPU pass\n{text}"
    );
}

#[test]
fn unknown_case_is_not_success() {
    let out = bin()
        .args(["--case", "not_a_real_case"])
        .output()
        .expect("run metal-correctness with unknown case");
    assert!(!out.status.success(), "unknown case must not succeed");
    let text = output_text(&out);
    assert!(text.contains("CASE_UNKNOWN"), "missing CASE_UNKNOWN\n{text}");
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
        yaml.contains("not issue 31 complete") || yaml.contains("not #31"),
        "workflow must say initial coverage is not issue 31 complete\n{yaml}"
    );
}
