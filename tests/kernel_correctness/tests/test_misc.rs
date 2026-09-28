use kernel_correctness::{broadcast, loss, math, optimizer};

const TOL: f32 = 1e-5;

fn assert_approx(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(err < TOL, "{ctx} elem {i}: got {g}, want {w}, err {err}");
    }
}

// ====================================================================
// Broadcast ops
// ====================================================================

#[test]
fn where_broadcast_basic() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [10.0, 20.0, 30.0, 40.0];
    let mask = [true, false, true, false];
    let mut output = [0.0; 4];
    broadcast::where_broadcast(&x, &y, &mask, &mut output);
    assert_eq!(output, [1.0, 20.0, 3.0, 40.0]);
}

#[test]
fn where_broadcast_all_true() {
    let x = [1.0, 2.0, 3.0];
    let y = [10.0, 20.0, 30.0];
    let mask = [true, true, true];
    let mut output = [0.0; 3];
    broadcast::where_broadcast(&x, &y, &mask, &mut output);
    assert_eq!(output, [1.0, 2.0, 3.0]);
}

#[test]
fn logic_and_broadcast_basic() {
    let a = [1.0, 0.0, 3.0, 0.0];
    let b = [5.0, 6.0, 0.0, 0.0];
    let mut output = [0.0; 4];
    broadcast::logic_and_broadcast(&a, &b, &mut output);
    assert_eq!(output, [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn power_broadcast_basic() {
    let base = [2.0, 3.0, 4.0];
    let exp = [3.0, 2.0, 0.5];
    let mut output = [0.0; 3];
    broadcast::power_broadcast(&base, &exp, &mut output);
    assert!((output[0] - 8.0).abs() < TOL, "2^3");
    assert!((output[1] - 9.0).abs() < TOL, "3^2");
    assert!((output[2] - 2.0).abs() < TOL, "4^0.5");
}

#[test]
fn power_broadcast_identity() {
    // x^1 = x
    let base = [1.5, 2.5, 3.5];
    let exp = [1.0, 1.0, 1.0];
    let mut output = [0.0; 3];
    broadcast::power_broadcast(&base, &exp, &mut output);
    assert_approx(&output, &base, "power_identity");
}

// ====================================================================
// Math ops
// ====================================================================

#[test]
fn masked_cumsum_basic() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mask = [true, false, true, true, false];
    let mut output = [0.0; 5];
    math::masked_cumsum(&input, &mask, &mut output);
    // acc: 1, 1, 4, 8, 8
    assert_approx(&output, &[1.0, 1.0, 4.0, 8.0, 8.0], "masked_cumsum");
}

#[test]
fn masked_cumsum_all_true() {
    let input = [1.0, 2.0, 3.0];
    let mask = [true, true, true];
    let mut output = [0.0; 3];
    math::masked_cumsum(&input, &mask, &mut output);
    assert_approx(&output, &[1.0, 3.0, 6.0], "masked_cumsum_all");
}

#[test]
fn masked_cumsum_all_false() {
    let input = [1.0, 2.0, 3.0];
    let mask = [false, false, false];
    let mut output = [0.0; 3];
    math::masked_cumsum(&input, &mask, &mut output);
    assert_approx(&output, &[0.0, 0.0, 0.0], "masked_cumsum_none");
}

// ====================================================================
// Loss ops
// ====================================================================

#[test]
fn triplet_margin_loss_positive() {
    // anchor close to positive, far from negative -> loss = 0
    let a = [0.0, 0.0];
    let p = [0.1, 0.1]; // ||a-p||^2 = 0.02
    let n = [10.0, 10.0]; // ||a-n||^2 = 200
    let loss = loss::triplet_margin_loss(&a, &p, &n, 1.0);
    // 0.02 - 200 + 1.0 = -198.98 -> max(0, -198.98) = 0
    assert!((loss - 0.0).abs() < TOL, "triplet_positive: loss={loss}");
}

#[test]
fn triplet_margin_loss_violation() {
    // anchor close to negative, far from positive -> loss > 0
    let a = [0.0, 0.0];
    let p = [10.0, 10.0]; // ||a-p||^2 = 200
    let n = [0.1, 0.1]; // ||a-n||^2 = 0.02
    let loss = loss::triplet_margin_loss(&a, &p, &n, 1.0);
    // 200 - 0.02 + 1.0 = 200.98
    assert!(
        (loss - 200.98).abs() < 0.01,
        "triplet_violation: loss={loss}"
    );
}

#[test]
fn triplet_margin_loss_margin_only() {
    // Same distance -> loss = margin
    let a = [0.0];
    let p = [1.0];
    let n = [1.0];
    let loss = loss::triplet_margin_loss(&a, &p, &n, 0.5);
    // ||a-p||^2 = 1, ||a-n||^2 = 1, loss = 1 - 1 + 0.5 = 0.5
    assert!((loss - 0.5).abs() < TOL, "triplet_margin: loss={loss}");
}

// ====================================================================
// Optimizer ops
// ====================================================================

#[test]
fn lamb_update_zero_grad() {
    // Zero gradient -> m and v stay (or grow toward) zero, param unchanged
    let mut param = [1.0, 2.0, 3.0];
    let grad = [0.0, 0.0, 0.0];
    let mut m = [0.0, 0.0, 0.0];
    let mut v = [0.0, 0.0, 0.0];
    optimizer::lamb_update(
        &mut param, &grad, &mut m, &mut v, 0.01, 0.9, 0.999, 1e-8, 0.9, 0.999,
    );
    // With zero grad, m=0, v=0, update=0/(0+eps)=0, param unchanged
    assert_approx(&param, &[1.0, 2.0, 3.0], "lamb_zero_grad");
}

#[test]
fn lamb_update_basic() {
    // Single step with non-zero gradient
    let mut param = [1.0, 2.0];
    let grad = [0.1, 0.2];
    let mut m = [0.0, 0.0];
    let mut v = [0.0, 0.0];
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let lr = 0.01f32;
    let eps = 1e-8f32;

    optimizer::lamb_update(
        &mut param, &grad, &mut m, &mut v, lr, beta1, beta2, eps, beta1, beta2,
    );

    // After one step, m and v should be updated
    assert!((m[0] - 0.01).abs() < 1e-6, "m[0]={}", m[0]); // (1-0.9)*0.1
    assert!((v[0] - 0.001 * 0.01).abs() < 1e-6, "v[0]={}", v[0]); // (1-0.999)*0.01
                                                                  // param should have decreased
    assert!(param[0] < 1.0, "param[0] should decrease: {}", param[0]);
    assert!(param[1] < 2.0, "param[1] should decrease: {}", param[1]);
}

#[test]
fn lamb_trust_ratio_with_zero_param() {
    // If param = [0,0,...], trust_ratio should be 1.0 (no scaling)
    let mut param = [0.0, 0.0];
    let grad = [1.0, 1.0];
    let mut m = [0.0, 0.0];
    let mut v = [0.0, 0.0];
    optimizer::lamb_update(
        &mut param, &grad, &mut m, &mut v, 0.01, 0.9, 0.999, 1e-8, 0.9, 0.999,
    );
    // Should not panic and param should become slightly negative
    assert!(param[0] < 0.0, "param should be negative: {}", param[0]);
}
