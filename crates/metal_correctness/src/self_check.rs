//! Comparator and completeness self-check. Injections never run on the GPU
//! and never become the CI Metal path.

use crate::cases::{
    delivered_ids, materialize, required_inputs_present, spec_by_id, DeviceCaps, RefOp,
};
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

fn hand_case(op: RefOp) -> crate::cases::Case {
    let id = format!("{}_f32_small", op.tag());
    let spec = spec_by_id(&id).unwrap_or_else(|| panic!("missing hand spec {id}"));
    materialize(&spec, None).unwrap_or_else(|e| panic!("materialize {id}: {e}"))
}

pub fn run() -> SelfCheck {
    let mut ok = true;
    let mut lines = Vec::new();

    // Every path's literal hand anchor must match an independent evaluation of
    // the declared computation, and must match the declared literal.
    for op in [RefOp::Add, RefOp::Sub, RefOp::Mul, RefOp::Exp, RefOp::AddMul] {
        let case = hand_case(op);
        let derived = op.apply(&case.inputs);
        let key = format!("SELF_CHECK_HAND_{}", op.tag().to_uppercase());
        let agrees = derived
            .iter()
            .zip(case.expected_anchor.iter())
            .all(|(a, b)| (a - b).abs() <= 1e-5 + 1.3e-6 * b.abs());
        if agrees {
            lines.push(format!("{key}=pass"));
        } else {
            ok = false;
            lines.push(format!("{key}=fail derived={derived:?} literal={:?}", case.expected_anchor));
        }
    }

    // The combination must expose wrong association: the literal wrong order
    // has to be rejected by the comparator against the correct anchor.
    let combo = hand_case(RefOp::AddMul);
    let wrong_order: Vec<f32> = combo.inputs[0]
        .iter()
        .zip(combo.inputs[1].iter())
        .zip(combo.inputs[2].iter())
        .map(|((a, b), c)| a + (b * c))
        .collect();
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_COMBO_ORDER",
        would_reject(&wrong_order, &combo.expected_anchor),
        "a+(b*c) was accepted for a (a+b)*c case",
    );

    let base = hand_case(RefOp::Add);
    let expected = base.expected_anchor.clone();

    // Obvious numerical mismatch on the last element.
    let mut bad = expected.clone();
    let last = bad.len() - 1;
    bad[last] = 99.0;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_MISMATCH",
        would_reject(&bad, &expected),
        "comparator accepted an obvious mismatch",
    );

    // All-zero output against non-zero expected: must not be indistinguishable.
    let zeros = vec![0.0f32; expected.len()];
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_ZERO_OUTPUT",
        would_reject(&zeros, &expected),
        "all-zero output was accepted",
    );

    // NaN / Inf.
    let mut nan = expected.clone();
    nan[0] = f32::NAN;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_NAN",
        would_reject(&nan, &expected),
        "NaN was accepted",
    );
    let mut inf = expected.clone();
    inf[1] = f32::INFINITY;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_INF",
        would_reject(&inf, &expected),
        "Inf was accepted",
    );

    // A tail-only miss must be caught (not just an interior mismatch).
    let mut tail_bad = expected.clone();
    tail_bad[expected.len() - 1] = expected[expected.len() - 1] + 10.0;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_TAIL_MISS",
        would_reject(&tail_bad, &expected),
        "tail miss was accepted",
    );

    // Shape / dtype.
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_SHAPE",
        require_shape(&[expected.len() - 1], &base.shape).is_err(),
        "shape mismatch was accepted",
    );
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_DTYPE",
        require_dtype("f16", base.dtype).is_err(),
        "dtype mismatch was accepted",
    );

    // Preserved-region destruction.
    let orig = base.inputs[0].clone();
    let mut smashed = orig.clone();
    smashed[2] = 42.0;
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_PRESERVE",
        assert_preserved("input_0", &smashed, &orig).is_err(),
        "input smash was accepted",
    );

    // Completeness: expected list with no execution must not succeed.
    let summary = RunSummary {
        results: Vec::new(),
        expected: delivered_ids(),
        complete_list: true,
    };
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_MISSING_CASE",
        !summary.missing().is_empty() && !summary.overall().is_success(),
        "empty execution was treated as a complete pass",
    );

    // A dropped delivered case must be reported missing, not silently passed.
    let all = delivered_ids();
    let short = RunSummary {
        results: all
            .iter()
            .take(all.len() - 1)
            .map(|id| crate::status::CaseResult::pass(id.clone()))
            .collect(),
        expected: all.clone(),
        complete_list: true,
    };
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_DROPPED_CASE",
        short.missing().len() == 1 && !short.overall().is_success(),
        "a dropped delivered case was not reported missing",
    );

    // Required input missing: empty fixture buffers must not be treated as ready.
    let mut missing = base.clone();
    missing.inputs[0].clear();
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_MISSING_INPUT",
        required_inputs_present(&missing).is_err(),
        "empty required input was accepted",
    );

    // Device-relative size without a device is unverified, not a guessed size.
    let device_spec = spec_by_id("add_f32_vec").expect("device-relative spec");
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_NO_DEVICE_SIZE",
        materialize(&device_spec, None).is_err(),
        "device-relative size was guessed without a device",
    );
    let caps = DeviceCaps {
        simd_width: 17,
        threadgroup_max: 29,
    };
    let resolved = materialize(&device_spec, Some(&caps)).expect("device-relative spec resolves");
    reject(
        &mut ok,
        &mut lines,
        "SELF_CHECK_DEVICE_SIZE",
        resolved.n() == 17,
        "device-relative size did not follow the discovered caps",
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
