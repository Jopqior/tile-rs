use kernel_correctness::activation::*;
use kernel_correctness::golden::assert_approx;
use kernel_correctness::normalization;

const TOL: f32 = 1e-4;

// ===== multiblock_kernel.rs (16 kernels) =====
// Multiblock kernels use get_block_idx() to split work across AICore blocks.
// The underlying math per-block is identical to the base operations.

#[test]
fn test_relu_multiblock() {
    let x = [-3.0, -1.0, 0.0, 1.0, 3.0];
    let result = relu(&x);
    assert_approx(&result, &[0.0, 0.0, 0.0, 1.0, 3.0], TOL, "relu_mb");
}

#[test]
fn test_sigmoid_multiblock() {
    let x = [-3.0, 0.0, 3.0];
    let result = sigmoid(&x);
    assert!((result[1] - 0.5).abs() < TOL);
    assert!(result[0] < 0.1);
    assert!(result[2] > 0.9);
}

#[test]
fn test_gelu_multiblock() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let result = gelu(&x);
    assert!((result[1] - 0.0).abs() < TOL);
    assert!(result[2] > 0.0);
    assert!(result[0] < 0.0);
}

#[test]
fn test_tanh_multiblock() {
    let x = [-3.0, 0.0, 3.0];
    let result = tanh_act(&x);
    assert!((result[1] - 0.0).abs() < TOL);
    assert!(result[0] < -0.9);
    assert!(result[2] > 0.9);
}

#[test]
fn test_softmax_multiblock() {
    let x = [1.0, 2.0, 3.0];
    let result = softmax(&x);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_layernorm_multiblock() {
    let x = [1.0, 3.0, 5.0, 7.0];
    let result = normalization::layernorm(&x, 1e-5);
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 0.01);
}

#[test]
fn test_vec_add_multiblock() {
    let x = [1.0, 2.0, 3.0];
    let y = [4.0, 5.0, 6.0];
    let result: Vec<f32> = x.iter().zip(&y).map(|(&a, &b)| a + b).collect();
    assert_approx(&result, &[5.0, 7.0, 9.0], TOL, "vec_add_mb");
}

#[test]
fn test_mish_multiblock() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let result = mish(&x);
    assert!((result[1] - 0.0).abs() < TOL);
    assert!(result[2] > 0.0);
}

#[test]
fn test_swish_multiblock() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let result = swish(&x);
    assert!((result[1] - 0.0).abs() < TOL);
    assert!(result[2] > 0.0);
}

#[test]
fn test_elu_multiblock() {
    let x = [-2.0, 0.0, 2.0];
    let result = elu(&x, 1.0);
    assert!((result[1] - 0.0).abs() < TOL);
    assert!(result[0] < 0.0 && result[0] > -1.0);
    assert!((result[2] - 2.0).abs() < TOL);
}

#[test]
fn test_selu_multiblock() {
    let x = [-1.0, 0.0, 1.0];
    let result = selu(&x);
    assert!(result[0] < 0.0);
    assert!((result[1] - 0.0).abs() < TOL);
    assert!((result[2] - 1.0507).abs() < 0.01);
}

#[test]
fn test_leaky_relu_multiblock() {
    let x = [-10.0, 0.0, 10.0];
    let result = leaky_relu(&x, 0.01);
    assert!((result[0] - (-0.1)).abs() < TOL);
    assert!((result[1] - 0.0).abs() < TOL);
    assert!((result[2] - 10.0).abs() < TOL);
}

#[test]
fn test_rmsnorm_multiblock() {
    let x = [2.0, 4.0, 6.0, 8.0];
    let result = normalization::rms_norm(&x, 1e-5);
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let expected: Vec<f32> = x.iter().map(|&v| v / rms).collect();
    assert_approx(&result, &expected, TOL, "rmsnorm_mb");
}

#[test]
fn test_hardswish_multiblock() {
    let x = [-4.0, 0.0, 3.0, 5.0];
    let result = hardswish(&x);
    assert!((result[0] - 0.0).abs() < TOL);
    assert!((result[1] - 0.0).abs() < TOL);
    assert!((result[2] - 3.0).abs() < TOL);
    assert!((result[3] - 5.0).abs() < TOL);
}

#[test]
fn test_hardsigmoid_multiblock() {
    let x = [-4.0, 0.0, 4.0];
    let result = hardsigmoid(&x);
    assert!((result[0] - 0.0).abs() < TOL);
    assert!((result[1] - 0.5).abs() < TOL);
    assert!((result[2] - 1.0).abs() < TOL);
}

#[test]
fn test_softplus_multiblock() {
    let x = [-2.0, 0.0, 2.0];
    let result = softplus(&x);
    assert!(
        (result[1] - (2.0f32).ln()).abs() < TOL,
        "softplus(0) = ln(2)"
    );
    assert!(result.iter().all(|&v| v > 0.0));
}
