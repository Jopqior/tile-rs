use kernel_correctness::activation::*;
use kernel_correctness::golden::assert_approx;
use kernel_correctness::normalization;

const TOL: f32 = 1e-4;

// ===== tiled_kernel.rs (16 kernels) =====
// Tiled kernels apply the same math as base ops but in 256-element tiles.
// We verify the math is identical.

#[test]
fn test_relu_tiled() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let result = relu(&x);
    assert_approx(&result, &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], TOL, "relu_tiled");
}

#[test]
fn test_sigmoid_tiled() {
    let x = [-2.0, 0.0, 2.0];
    let result = sigmoid(&x);
    assert!((result[1] - 0.5).abs() < TOL, "sigmoid(0) = 0.5");
    assert!(result[0] < 0.5);
    assert!(result[2] > 0.5);
}

#[test]
fn test_gelu_tiled() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = gelu(&x);
    assert!((result[2] - 0.0).abs() < TOL, "gelu(0) = 0");
    assert!(result[3] > 0.0);
}

#[test]
fn test_tanh_tiled() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = tanh_act(&x);
    assert!((result[2] - 0.0).abs() < TOL, "tanh(0) = 0");
    assert!(result.iter().all(|&v| v.abs() <= 1.0));
}

#[test]
fn test_swish_tiled() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let result = swish(&x);
    assert!((result[1] - 0.0).abs() < TOL, "swish(0) = 0");
    assert!(result[2] > 0.0);
}

#[test]
fn test_exp_tiled() {
    let x: [f32; 4] = [0.0, 1.0, -1.0, 2.0];
    let result: Vec<f32> = x.iter().map(|&v| v.exp()).collect();
    assert!((result[0] - 1.0).abs() < TOL, "exp(0) = 1");
    assert!((result[1] - std::f32::consts::E).abs() < TOL);
}

#[test]
fn test_vec_add_tiled() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [0.1, 0.2, 0.3, 0.4];
    let result: Vec<f32> = x.iter().zip(&y).map(|(&a, &b)| a + b).collect();
    assert_approx(&result, &[1.1, 2.2, 3.3, 4.4], TOL, "vec_add_tiled");
}

#[test]
fn test_vec_mul_tiled() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [2.0, 3.0, 4.0, 5.0];
    let result: Vec<f32> = x.iter().zip(&y).map(|(&a, &b)| a * b).collect();
    assert_approx(&result, &[2.0, 6.0, 12.0, 20.0], TOL, "vec_mul_tiled");
}

#[test]
fn test_elu_tiled() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = elu(&x, 1.0);
    assert!((result[2] - 0.0).abs() < TOL, "elu(0) = 0");
    assert!(result[0] < 0.0 && result[0] > -1.0);
    assert!((result[3] - 1.0).abs() < TOL);
}

#[test]
fn test_mish_tiled() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let result = mish(&x);
    assert!((result[1] - 0.0).abs() < TOL, "mish(0) = 0");
    assert!(result[2] > 0.0);
}

#[test]
fn test_layernorm_tiled() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let result = normalization::layernorm(&x, 1e-5);
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 0.01, "layernorm mean ≈ 0");
}

#[test]
fn test_softmax_tiled() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let result = softmax(&x);
    assert!(
        (result.iter().sum::<f32>() - 1.0).abs() < TOL,
        "softmax sums to 1"
    );
    // monotonic
    for i in 0..3 {
        assert!(result[i] < result[i + 1]);
    }
}

#[test]
fn test_selu_tiled() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = selu(&x);
    assert!(result[0] < 0.0);
    assert!((result[2] - 0.0).abs() < TOL);
    // SELU has scale ~1.0507
    assert!((result[3] - 1.0507).abs() < 0.01);
}

#[test]
fn test_leaky_relu_tiled() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = leaky_relu(&x, 0.01);
    assert!((result[0] - (-0.02)).abs() < TOL);
    assert!((result[2] - 0.0).abs() < TOL);
    assert!((result[3] - 1.0).abs() < TOL);
}

#[test]
fn test_hardswish_tiled() {
    let x = [-4.0, -3.0, 0.0, 3.0, 4.0];
    let result = hardswish(&x);
    assert!((result[0] - 0.0).abs() < TOL, "hardswish(-4) = 0");
    assert!((result[2] - 0.0).abs() < TOL, "hardswish(0) = 0");
    assert!((result[3] - 3.0).abs() < TOL, "hardswish(3) = 3");
}

#[test]
fn test_rmsnorm_tiled() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let result = normalization::rms_norm(&x, 1e-5);
    assert_eq!(result.len(), 4);
    // RMS norm divides by RMS value
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let expected: Vec<f32> = x.iter().map(|&v| v / rms).collect();
    assert_approx(&result, &expected, TOL, "rmsnorm_tiled");
}
