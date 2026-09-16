//! Comparator and completeness self-check. Injections never run on the GPU
//! and never become the CI Metal path.

use crate::cases::{add_f32_small, delivered_ids, ADD_F32_SMALL_HAND};
use crate::compare::{assert_close, assert_preserved, require_dtype, require_shape, would_reject};
use crate::status::{RunSummary, Status};

pub struct SelfCheck {
    pub ok: bool,
    pub lines: Vec<String>,
}

fn reject(ok: &mut bool, lines: &mut Vec<String>, key: &str, caught: bool, detail: &str) {
    if caught {
        lines.push(format!("{key}=rejected"));
    } else {
        *ok = false;
        lines.push(format!("{key}=NOT_REJECTED {detail}"));
    }
}

pub fn run() -> SelfCheck {
    let mut ok = true;
    let mut lines = Vec::new();
    let case = add_f32_small();

    // Hand anchors must be the independently known literals for this case.
    if case.hand_expected != ADD_F32_SMALL_HAND {
        ok = false;
        lines.push("SELF_CHECK_HAND=fail hand literals drifted from the case".into());
    } else {
        lines.push("SELF_CHECK_HAND=pass".into());
    }

    // Obvious numerical mismatch (last element 99 vs -0.25).
    let mut bad = ADD_F32_SMALL_HAND.to_vec();
    bad[7] = 99.0;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_MISMATCH",
        would_reject(&bad, &ADD_F32_SMALL_HAND),
        "comparator accepted actual[-0.25] vs 99",
    );

    // All-zero output against non-zero expected: must not be indistinguishable.
    let zeros = vec![0.0f32; ADD_F32_SMALL_HAND.len()];
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_ZERO_OUTPUT",
        would_reject(&zeros, &ADD_F32_SMALL_HAND),
        "all-zero output was accepted",
    );

    // NaN / Inf.
    let mut nan = ADD_F32_SMALL_HAND.to_vec();
    nan[0] = f32::NAN;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_NAN",
        would_reject(&nan, &ADD_F32_SMALL_HAND),
        "NaN was accepted",
    );
    let mut inf = ADD_F32_SMALL_HAND.to_vec();
    inf[1] = f32::INFINITY;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_INF",
        would_reject(&inf, &ADD_F32_SMALL_HAND),
        "Inf was accepted",
    );

    // Shape / dtype.
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_SHAPE",
        require_shape(&[7], case.shape).is_err(),
        "shape mismatch was accepted",
    );
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_DTYPE",
        require_dtype("f16", case.dtype).is_err(),
        "dtype mismatch was accepted",
    );

    // Preserved-region destruction.
    let orig = case.a.to_vec();
    let mut smashed = orig.clone();
    smashed[2] = 42.0;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_PRESERVE",
        assert_preserved("input_a", &smashed, &orig).is_err(),
        "input smash was accepted",
    );

    // Completeness: expected list with no execution must not succeed.
    let summary = RunSummary {
        results: Vec::new(),
        expected: delivered_ids().into_iter().map(|s| s.to_string()).collect(),
        complete_list: true,
    };
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_MISSING_CASE",
        !summary.missing().is_empty() && !summary.overall().is_success(),
        "empty execution was treated as a complete pass",
    );

    // Required input missing.
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_MISSING_INPUT",
        assert_close(&[], &ADD_F32_SMALL_HAND).is_err(),
        "empty actual was accepted",
    );

    // Device unavailable is unverified, not pass.
    let unverified = Status::Unverified;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_DEVICE_UNAVAILABLE",
        !unverified.is_success(),
        "unverified was treated as pass",
    );

    if ok {
        lines.push("SELF_CHECK=pass".into());
    } else {
        lines.push("SELF_CHECK=fail".into());
    }
    SelfCheck { ok, lines }
}
