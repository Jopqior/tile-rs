use kernel_correctness::golden::assert_approx;
use kernel_correctness::normalization::*;

const TOL: f32 = 1e-5;

// ===== Layer norm (norm_ops + layernorm_kernel) =====
#[test]
fn test_layernorm_uniform() {
    assert_approx(
        &layernorm(&[3.0, 3.0, 3.0, 3.0], 1e-5),
        &[0.0, 0.0, 0.0, 0.0],
        TOL,
        "ln_uniform",
    );
}
#[test]
fn test_layernorm_known() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let mean = 2.5f32;
    let var = 1.25f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    let expected: Vec<f32> = input.iter().map(|x| (x - mean) * inv_std).collect();
    assert_approx(&layernorm(&input, 1e-5), &expected, TOL, "ln_known");
}
#[test]
fn test_layernorm_two() {
    let out = layernorm(&[-1.0, 1.0], 1e-5);
    assert_approx(&out, &[-1.0, 1.0], 1e-3, "ln_two");
}

// ===== RMS norm =====
#[test]
fn test_rms_norm_ones() {
    assert_approx(
        &rms_norm(&[1.0, 1.0, 1.0, 1.0], 1e-5),
        &[1.0, 1.0, 1.0, 1.0],
        1e-3,
        "rms_ones",
    );
}
#[test]
fn test_rms_norm_basic() {
    let input = [3.0, 4.0];
    let rms = (12.5f32 + 1e-5).sqrt();
    assert_approx(
        &rms_norm(&input, 1e-5),
        &[3.0 / rms, 4.0 / rms],
        TOL,
        "rms_basic",
    );
}

// ===== L1/L2 norm =====
#[test]
fn test_l1_norm() {
    assert!((l1_norm(&[-3.0, 4.0]) - 7.0).abs() < TOL);
}
#[test]
fn test_l2_norm() {
    assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < TOL);
}
#[test]
fn test_l2_normalize() {
    let out = l2_normalize(&[3.0, 4.0], 1e-8);
    assert!((out[0] - 0.6).abs() < TOL && (out[1] - 0.8).abs() < TOL);
}
#[test]
fn test_frobenius() {
    assert!((frobenius_norm(&[3.0, 4.0]) - 5.0).abs() < TOL);
}

// ===== Batch norm (norm_extended_kernel) =====
#[test]
fn test_batch_norm_identity() {
    // x=input, mean=0 vec, var=1 vec => (x-0)/sqrt(1+eps) ≈ x
    let out = batch_norm_vec(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], 1e-5);
    assert_approx(&out, &[1.0, 2.0, 3.0], 1e-3, "bn_identity");
}

// ===== Group norm =====
#[test]
fn test_group_norm() {
    let out = group_norm(&[1.0, 2.0, 3.0, 4.0], 1, 1e-5);
    let expected = layernorm(&[1.0, 2.0, 3.0, 4.0], 1e-5);
    assert_approx(&out, &expected, TOL, "gn_single");
}

// ===== Instance norm =====
#[test]
fn test_instance_norm() {
    let out = instance_norm(&[1.0, 3.0], 1e-5);
    let expected = layernorm(&[1.0, 3.0], 1e-5);
    assert_approx(&out, &expected, TOL, "instnorm");
}
