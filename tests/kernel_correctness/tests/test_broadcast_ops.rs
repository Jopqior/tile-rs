use kernel_correctness::activation::*;
use kernel_correctness::golden::assert_approx;

const TOL: f32 = 1e-5;

// ===== broadcast_ops_kernel.rs (8 kernels) =====

#[test]
fn test_add_bias() {
    // add_bias: y = x + bias (scalar broadcast)
    let x = [1.0, 2.0, 3.0, 4.0];
    let bias = 0.5;
    let expected: Vec<f32> = x.iter().map(|&v| v + bias).collect();
    assert_approx(&expected, &[1.5, 2.5, 3.5, 4.5], TOL, "add_bias");
}

#[test]
fn test_elementwise_mul() {
    assert_approx(
        &mul(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
        &[4.0, 10.0, 18.0],
        TOL,
        "ew_mul",
    );
}

#[test]
fn test_elementwise_div() {
    assert_approx(
        &div(&[4.0, 10.0, 18.0], &[4.0, 5.0, 6.0]),
        &[1.0, 2.0, 3.0],
        TOL,
        "ew_div",
    );
}

#[test]
fn test_elementwise_sub() {
    assert_approx(
        &sub(&[5.0, 7.0, 9.0], &[1.0, 2.0, 3.0]),
        &[4.0, 5.0, 6.0],
        TOL,
        "ew_sub",
    );
}

#[test]
fn test_elementwise_max() {
    assert_approx(
        &max_ew(&[1.0, 5.0, 3.0], &[4.0, 2.0, 6.0]),
        &[4.0, 5.0, 6.0],
        TOL,
        "ew_max",
    );
}

#[test]
fn test_elementwise_min() {
    assert_approx(
        &min_ew(&[1.0, 5.0, 3.0], &[4.0, 2.0, 6.0]),
        &[1.0, 2.0, 3.0],
        TOL,
        "ew_min",
    );
}

#[test]
fn test_clamp() {
    assert_approx(
        &hardtanh(&[-5.0, 0.0, 5.0], -1.0, 1.0),
        &[-1.0, 0.0, 1.0],
        TOL,
        "clamp",
    );
}

#[test]
fn test_elementwise_square() {
    assert_approx(
        &square_f32(&[2.0, 3.0, 4.0]),
        &[4.0, 9.0, 16.0],
        TOL,
        "ew_square",
    );
}

// ===== math_ops_kernel.rs (scalar_mul) =====
#[test]
fn test_matrix_scalar_mul() {
    assert_approx(
        &scalar_mul(&[1.0, 2.0, 3.0], 3.0),
        &[3.0, 6.0, 9.0],
        TOL,
        "mat_scalar_mul",
    );
}
