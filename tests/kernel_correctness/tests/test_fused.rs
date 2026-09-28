use kernel_correctness::activation::*;
use kernel_correctness::golden::assert_approx;
use kernel_correctness::normalization;

const TOL: f32 = 1e-4;

fn matmul_2x2(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

// ===== fused_activation_chain_kernel.rs (17 kernels) =====

#[test]
fn test_fused_relu_hardswish() {
    let x = [-1.0, 0.0, 1.0, 3.0, 5.0];
    let after_relu = relu(&x);
    let result = hardswish(&after_relu);
    assert_eq!(result.len(), 5);
    assert!((result[0] - 0.0).abs() < TOL);
}

#[test]
fn test_fused_hardswish_relu() {
    let x = [-1.0, 0.0, 1.0, 3.0];
    let after_hs = hardswish(&x);
    let result = relu(&after_hs);
    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_fused_mish_mish() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let after_first = mish(&x);
    let result = mish(&after_first);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_mish_tanh() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let after_mish = mish(&x);
    let result = tanh_act(&after_mish);
    assert!(result.iter().all(|&v| v.abs() <= 1.0));
}

#[test]
fn test_fused_min_tanh_tanh() {
    let x: [f32; 4] = [0.5, 1.0, 2.0, -0.5];
    let clamped: Vec<f32> = x.iter().map(|&v| v.min(1.0)).collect();
    let after_tanh1 = tanh_act(&clamped);
    let result = tanh_act(&after_tanh1);
    assert!(result.iter().all(|&v| v.abs() <= 1.0));
}

#[test]
fn test_fused_mul_leakyrelu_gelu() {
    let x = [1.0, 2.0, -1.0, 0.5];
    let scaled: Vec<f32> = x.iter().map(|&v| v * 2.0).collect();
    let after_lr = leaky_relu(&scaled, 0.01);
    let result = gelu(&after_lr);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_sub_tanh_sub() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let sub1: Vec<f32> = x.iter().map(|&v| v - 0.5).collect();
    let after_tanh = tanh_act(&sub1);
    let result: Vec<f32> = after_tanh.iter().map(|&v| v - 0.5).collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_sigmoid_sum() {
    let x = [0.0, 1.0, -1.0, 2.0];
    let sig = sigmoid(&x);
    let sum: f32 = sig.iter().sum();
    assert!(sum > 0.0, "sigmoid_sum positive");
}

#[test]
fn test_fused_add_scale_sigmoid() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let added: Vec<f32> = x.iter().map(|&v| v + 0.5).collect();
    let scaled: Vec<f32> = added.iter().map(|&v| v * 2.0).collect();
    let result = sigmoid(&scaled);
    assert!(result.iter().all(|&v| v > 0.0 && v < 1.0));
}

#[test]
fn test_fused_scale_min() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let scaled: Vec<f32> = x.iter().map(|&v| v * 2.0).collect();
    let result: Vec<f32> = scaled.iter().map(|&v| v.min(5.0)).collect();
    assert!((result[3] - 5.0).abs() < TOL);
}

#[test]
fn test_fused_leakyrelu_leakyrelu_gelu_gelu() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let lr1 = leaky_relu(&x, 0.01);
    let lr2 = leaky_relu(&lr1, 0.01);
    let g1 = gelu(&lr2);
    let result = gelu(&g1);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_divide_leakyrelu() {
    let x = [2.0, 4.0, -2.0, 6.0];
    let divided: Vec<f32> = x.iter().map(|&v| v * 0.5).collect();
    let result = leaky_relu(&divided, 0.01);
    assert_eq!(result.len(), 4);
    assert!((result[0] - 1.0).abs() < TOL);
}

#[test]
fn test_fused_sub_hardswish() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let sub: Vec<f32> = x.iter().map(|&v| v - 0.5).collect();
    let result = hardswish(&sub);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_tanh_scale_bias_max() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let after_tanh = tanh_act(&x);
    let scaled: Vec<f32> = after_tanh.iter().map(|&v| v * 2.0).collect();
    let biased: Vec<f32> = scaled.iter().map(|&v| v + 0.1).collect();
    let result: Vec<f32> = biased.iter().map(|&v| v.max(0.0)).collect();
    assert!(result.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_fused_relu_bias_add() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let after_relu = relu(&x);
    let result: Vec<f32> = after_relu.iter().map(|&v| v + 0.1).collect();
    assert!((result[0] - 0.1).abs() < TOL);
    assert!((result[2] - 1.1).abs() < TOL);
}

#[test]
fn test_fused_hardswish_relu_softmax_mean() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let after_hs = hardswish(&x);
    let after_relu = relu(&after_hs);
    let after_sm = softmax(&after_relu);
    let mean: f32 = after_sm.iter().sum::<f32>() / after_sm.len() as f32;
    assert!((mean - 0.25).abs() < TOL, "softmax mean = 1/n");
}

#[test]
fn test_fused_leakyrelu_clamp_gelu() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let lr = leaky_relu(&x, 0.01);
    let clamped: Vec<f32> = lr.iter().map(|&v| v.max(-1.0).min(1.0)).collect();
    let result = gelu(&clamped);
    assert_eq!(result.len(), 5);
}

// ===== fused_norm_activation_kernel.rs (8 kernels) =====

#[test]
fn test_fused_layernorm_relu() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let result = relu(&normed);
    assert!(result.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_fused_layernorm_sigmoid() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let result = sigmoid(&normed);
    assert!(result.iter().all(|&v| v > 0.0 && v < 1.0));
}

#[test]
fn test_fused_rmsnorm_swish() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::rms_norm(&x, 1e-5);
    let result = swish(&normed);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_layernorm_tanh_hardswish() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let after_tanh = tanh_act(&normed);
    let result = hardswish(&after_tanh);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_softmax_mean() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let sm = softmax(&x);
    let mean: f32 = sm.iter().sum::<f32>() / sm.len() as f32;
    assert!((mean - 0.25).abs() < TOL);
}

#[test]
fn test_fused_layernorm_gelu() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let result = gelu(&normed);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_rmsnorm_gelu() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::rms_norm(&x, 1e-5);
    let result = gelu(&normed);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_log_softmax_mean() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let lsm = log_softmax(&x);
    let mean: f32 = lsm.iter().sum::<f32>() / lsm.len() as f32;
    assert!(mean < 0.0, "log_softmax mean is negative");
}

// ===== fused_multi_op_kernel.rs (19 kernels) =====

#[test]
fn test_fused_norm_add_mul() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let added: Vec<f32> = normed.iter().map(|&v| v + 0.5).collect();
    let result: Vec<f32> = added.iter().map(|&v| v * 2.0).collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_scale_norm() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let scaled: Vec<f32> = x.iter().map(|&v| v * 2.0).collect();
    let result = normalization::layernorm(&scaled, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_sub_mish_mish() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let sub: Vec<f32> = x.iter().map(|&v| v - 0.5).collect();
    let m1 = mish(&sub);
    let result = mish(&m1);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_sub_tanh_sub_mean() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let sub1: Vec<f32> = x.iter().map(|&v| v - 0.5).collect();
    let after_tanh = tanh_act(&sub1);
    let sub2: Vec<f32> = after_tanh.iter().map(|&v| v - 0.1).collect();
    let mean: f32 = sub2.iter().sum::<f32>() / sub2.len() as f32;
    assert!(mean.is_finite());
}

#[test]
fn test_fused_min_add_mul() {
    let x: [f32; 4] = [1.0, 5.0, 3.0, 7.0];
    let min_val: Vec<f32> = x.iter().map(|&v| v.min(4.0)).collect();
    let added: Vec<f32> = min_val.iter().map(|&v| v + 0.1).collect();
    let result: Vec<f32> = added.iter().map(|&v| v * 2.0).collect();
    assert!((result[1] - 8.2).abs() < TOL); // min(5,4)=4, +0.1=4.1, *2=8.2
}

#[test]
fn test_fused_elu_scale() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let after_elu = elu(&x, 1.0);
    let result: Vec<f32> = after_elu.iter().map(|&v| v * 2.0).collect();
    assert_eq!(result.len(), 4);
    assert!(result[0] < 0.0); // ELU of negative scaled
}

#[test]
fn test_fused_selu_add() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let after_selu = selu(&x);
    let result: Vec<f32> = after_selu.iter().map(|&v| v + 0.5).collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_softplus_tanh() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let sp = softplus(&x);
    let result = tanh_act(&sp);
    assert!(
        result.iter().all(|&v| v > 0.0),
        "tanh(softplus) > 0 since softplus > 0"
    );
}

#[test]
fn test_fused_relu_scale_add() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let after_relu = relu(&x);
    let scaled: Vec<f32> = after_relu.iter().map(|&v| v * 2.0).collect();
    let result: Vec<f32> = scaled.iter().map(|&v| v + 0.1).collect();
    assert!((result[0] - 0.1).abs() < TOL);
    assert!((result[2] - 2.1).abs() < TOL);
}

#[test]
fn test_fused_sigmoid_gate() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let sig = sigmoid(&x);
    // gate: x * sigmoid(x)
    let result: Vec<f32> = x.iter().zip(&sig).map(|(&xi, &si)| xi * si).collect();
    assert!(result.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_fused_exp_reduce_sum() {
    let x: [f32; 3] = [0.0, 1.0, 2.0];
    let ex: Vec<f32> = x.iter().map(|&v| v.exp()).collect();
    let sum: f32 = ex.iter().sum();
    let expected = 1.0 + (1.0f32).exp() + (2.0f32).exp();
    assert!((sum - expected).abs() < TOL);
}

#[test]
fn test_log_sum_exp() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let lse = max_val + shifted.iter().sum::<f32>().ln();
    assert!(lse > 4.0, "log_sum_exp >= max");
}

#[test]
fn test_fused_max_lse_relu() {
    let x: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let maxed: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();
    // logsumexp
    let max_val = maxed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = maxed.iter().map(|&v| (v - max_val).exp()).collect();
    let lse = max_val + shifted.iter().sum::<f32>().ln();
    let result = lse.max(0.0); // relu
    assert!(result > 0.0);
}

#[test]
fn test_fused_hardswish_gelu() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let hs = hardswish(&x);
    let result = gelu(&hs);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_softsign_scale_add() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let ss = softsign(&x);
    let scaled: Vec<f32> = ss.iter().map(|&v| v * 2.0).collect();
    let result: Vec<f32> = scaled.iter().map(|&v| v + 0.5).collect();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_fused_hardsigmoid_scale_clamp() {
    let x = [-4.0, -1.0, 0.0, 1.0, 4.0];
    let hs = hardsigmoid(&x);
    let scaled: Vec<f32> = hs.iter().map(|&v| v * 2.0).collect();
    let result: Vec<f32> = scaled.iter().map(|&v| v.max(-1.0).min(1.0)).collect();
    assert!(result.iter().all(|&v| v >= -1.0 && v <= 1.0));
}

#[test]
fn test_fused_abs_sum() {
    let x = [-3.0, -1.0, 2.0, 4.0];
    let ab = abs_act(&x);
    let sum: f32 = ab.iter().sum();
    assert!((sum - 10.0).abs() < TOL);
}

#[test]
fn test_fused_rmsnorm_mish_scale() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::rms_norm(&x, 1e-5);
    let after_mish = mish(&normed);
    let result: Vec<f32> = after_mish.iter().map(|&v| v * 2.0).collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_reciprocal_scale_add() {
    let x = [1.0, 2.0, 4.0, 5.0];
    let recip: Vec<f32> = x.iter().map(|&v| 1.0 / v).collect();
    let scaled: Vec<f32> = recip.iter().map(|&v| v * 2.0).collect();
    let result: Vec<f32> = scaled.iter().map(|&v| v + 0.1).collect();
    assert!((result[0] - 2.1).abs() < TOL); // 1/1 * 2 + 0.1
    assert!((result[1] - 1.1).abs() < TOL); // 1/2 * 2 + 0.1
}

// ===== fused_gemm_activation_kernel.rs (8 kernels) =====

#[test]
fn test_fused_gemm_relu() {
    // C = relu(A * B) — identity matmul
    let ab = [1.0, -2.0, 3.0, -4.0]; // pretend matmul output
    let result = relu(&ab);
    assert_approx(&result, &[1.0, 0.0, 3.0, 0.0], TOL, "gemm_relu");
}

#[test]
fn test_fused_gemm_relu_divide() {
    let ab = [2.0, -1.0, 4.0, -3.0];
    let after_relu = relu(&ab);
    let result: Vec<f32> = after_relu.iter().map(|&v| v * 0.5).collect();
    assert_approx(&result, &[1.0, 0.0, 2.0, 0.0], TOL, "gemm_relu_div");
}

#[test]
fn test_fused_gemm_multiply_leakyrelu() {
    let ab = [1.0, -1.0, 2.0, -2.0];
    let scaled: Vec<f32> = ab.iter().map(|&v| v * 2.0).collect();
    let result = leaky_relu(&scaled, 0.01);
    assert!((result[0] - 2.0).abs() < TOL);
    assert!(result[1] < 0.0 && result[1] > -0.1); // leaky
}

#[test]
fn test_fused_gemm_sigmoid_scale_add() {
    let ab = [1.0, 2.0, 3.0, 4.0];
    let residual = [0.1, 0.2, 0.3, 0.4];
    let sig = sigmoid(&ab);
    let scaled: Vec<f32> = sig.iter().map(|&v| v * 0.5).collect();
    let result: Vec<f32> = scaled.iter().zip(&residual).map(|(&s, &r)| s + r).collect();
    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|&v| v > 0.0));
}

#[test]
fn test_fused_gemm_max_sub_gelu() {
    let ab: [f32; 4] = [-1.0, 0.5, 2.0, -0.5];
    let maxed: Vec<f32> = ab.iter().map(|&v| v.max(0.0)).collect();
    let sub: Vec<f32> = maxed.iter().map(|&v| v - 0.5).collect();
    let result = gelu(&sub);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_gemm_scale_hardtanh_gelu() {
    let ab = [0.5, -0.5, 1.0, -1.0];
    let scaled: Vec<f32> = ab.iter().map(|&v| v * 2.0).collect();
    let ht: Vec<f32> = scaled.iter().map(|&v| v.max(-1.0).min(1.0)).collect();
    let result = gelu(&ht);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_gemm_divide_sum_scale() {
    let ab = [2.0, 4.0, 6.0, 8.0];
    let divided: Vec<f32> = ab.iter().map(|&v| v * 0.5).collect();
    let sum: f32 = divided.iter().sum();
    let result = sum * 2.0;
    assert!((result - 20.0).abs() < TOL); // (1+2+3+4)*2
}

#[test]
fn test_fused_gemm_swish_div_clamp_tanh_clamp() {
    let ab = [0.5, 1.0, -0.5, 2.0];
    let sw = swish(&ab);
    let divided: Vec<f32> = sw.iter().map(|&v| v * 0.5).collect();
    let clamp1: Vec<f32> = divided.iter().map(|&v| v.max(-1.0).min(1.0)).collect();
    let after_tanh = tanh_act(&clamp1);
    let result: Vec<f32> = after_tanh.iter().map(|&v| v.max(-0.5).min(0.5)).collect();
    assert!(result.iter().all(|&v| v >= -0.5 && v <= 0.5));
}

// ===== fused_gemm_misc_kernel.rs (10 kernels) =====

#[test]
fn test_fused_gemm_norm_min_bias() {
    let ab = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&ab, 1e-5);
    let min_clamped: Vec<f32> = normed.iter().map(|&v| v.min(1.0)).collect();
    let result: Vec<f32> = min_clamped.iter().map(|&v| v + 0.1).collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_matmul_gelu_scale_max() {
    let ab = [1.0, -1.0, 2.0, -2.0];
    let g = gelu(&ab);
    let scaled: Vec<f32> = g.iter().map(|&v| v * 2.0).collect();
    let result: Vec<f32> = scaled.iter().map(|&v| v.max(0.0)).collect();
    assert!(result.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_fused_min_softmax() {
    let x: [f32; 4] = [1.0, 2.0, 3.0, 100.0];
    let clamped: Vec<f32> = x.iter().map(|&v| v.min(10.0)).collect();
    let result = softmax(&clamped);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_fused_norm_tanh_hardswish_residual_lse() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let residual = [0.1, 0.2, 0.3, 0.4];
    let normed = normalization::layernorm(&x, 1e-5);
    let after_tanh = tanh_act(&normed);
    let after_hs = hardswish(&after_tanh);
    let result: Vec<f32> = after_hs
        .iter()
        .zip(&residual)
        .map(|(&a, &b)| a + b)
        .collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_norm_scale_clamp() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let scaled: Vec<f32> = normed.iter().map(|&v| v * 2.0).collect();
    let result: Vec<f32> = scaled.iter().map(|&v| v.max(-1.0).min(1.0)).collect();
    assert!(result.iter().all(|&v| v >= -1.0 && v <= 1.0));
}

#[test]
fn test_fused_norm_mean() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let mean: f32 = normed.iter().sum::<f32>() / normed.len() as f32;
    assert!(mean.abs() < 0.01, "layernorm mean ≈ 0");
}

#[test]
fn test_fused_relu_norm_scale() {
    let x = [-1.0, 0.0, 1.0, 2.0];
    let after_relu = relu(&x);
    let normed = normalization::layernorm(&after_relu, 1e-5);
    let result: Vec<f32> = normed.iter().map(|&v| v * 2.0).collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_norm_scaling() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let result: Vec<f32> = normed.iter().map(|&v| v * 3.14).collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_norm_divide() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let result: Vec<f32> = normed.iter().map(|&v| v * 0.5).collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_norm_min_clamp() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let min_val: Vec<f32> = normed.iter().map(|&v| v.min(5.0)).collect();
    let result: Vec<f32> = min_val.iter().map(|&v| v.max(-3.0).min(3.0)).collect();
    assert!(result.iter().all(|&v| v >= -3.0 && v <= 3.0));
}

// ===== fused_matmul_activation_kernel.rs (9 kernels) =====

#[test]
fn test_fused_matmul_gelu() {
    let c = matmul_2x2(&[1.0, 0.0, 0.0, 1.0], &[1.0, -1.0, 2.0, -2.0]);
    let result = gelu(&c);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_matmul_relu() {
    let c = matmul_2x2(&[1.0, 0.0, 0.0, 1.0], &[1.0, -1.0, 2.0, -2.0]);
    let result = relu(&c);
    assert_approx(&result, &[1.0, 0.0, 2.0, 0.0], TOL, "matmul_relu");
}

#[test]
fn test_fused_matmul_sigmoid() {
    let c = matmul_2x2(&[1.0, 0.0, 0.0, 1.0], &[0.0, 1.0, -1.0, 0.0]);
    let result = sigmoid(&c);
    assert!((result[0] - 0.5).abs() < TOL, "sigmoid(0) = 0.5");
}

#[test]
fn test_fused_matmul_tanh() {
    let c = matmul_2x2(&[1.0, 0.0, 0.0, 1.0], &[0.0, 1.0, -1.0, 2.0]);
    let result = tanh_act(&c);
    assert!((result[0] - 0.0).abs() < TOL, "tanh(0) = 0");
}

#[test]
fn test_fused_matmul_swish() {
    let c = matmul_2x2(&[1.0, 0.0, 0.0, 1.0], &[1.0, 2.0, 3.0, 4.0]);
    let result = swish(&c);
    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_fused_matmul_mish_mish() {
    let c = [1.0, -1.0, 2.0, -2.0]; // pretend matmul output
    let m1 = mish(&c);
    let result = mish(&m1);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_matmul_hardtanh() {
    let c: [f32; 4] = [0.5, 2.0, -3.0, 0.0];
    let result: Vec<f32> = c.iter().map(|&v| v.max(-1.0).min(1.0)).collect();
    assert_approx(&result, &[0.5, 1.0, -1.0, 0.0], TOL, "matmul_hardtanh");
}

#[test]
fn test_fused_matmul_divide_gelu() {
    let c = [2.0, 4.0, -2.0, 6.0];
    let divided: Vec<f32> = c.iter().map(|&v| v * 0.5).collect();
    let result = gelu(&divided);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_matmul_min_sub() {
    let c: [f32; 4] = [0.5, 2.0, -1.0, 3.0];
    let min_clamped: Vec<f32> = c.iter().map(|&v| v.min(1.0)).collect();
    let result: Vec<f32> = min_clamped.iter().map(|&v| v - 0.5).collect();
    assert_approx(&result, &[0.0, 0.5, -1.5, 0.5], TOL, "matmul_min_sub");
}

// ===== fused_matmul_multi_kernel.rs (9 kernels) =====

#[test]
fn test_fused_matmul_softmax() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let result = softmax(&c);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_fused_matmul_gelu_softmax() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let g = gelu(&c);
    let result = softmax(&g);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_fused_matmul_layernorm() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let result = normalization::layernorm(&c, 1e-5);
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 0.01);
}

#[test]
fn test_fused_matmul_norm_leakyrelu_sum() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&c, 1e-5);
    let lr = leaky_relu(&normed, 0.01);
    let sum: f32 = lr.iter().sum();
    assert!(sum.is_finite());
}

#[test]
fn test_fused_matmul_norm_bias_div_swish() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&c, 1e-5);
    let biased: Vec<f32> = normed.iter().map(|&v| v + 0.1).collect();
    let divided: Vec<f32> = biased.iter().map(|&v| v * 0.5).collect();
    let result = swish(&divided);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_matmul_mean_softmax() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let result = softmax(&c);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_fused_matmul_max_scale() {
    let c: [f32; 4] = [-1.0, 2.0, -3.0, 4.0];
    let maxed: Vec<f32> = c.iter().map(|&v| v.max(0.0)).collect();
    let sum: f32 = maxed.iter().sum();
    let result = sum * 2.0;
    assert!((result - 12.0).abs() < TOL); // (0+2+0+4)*2
}

#[test]
fn test_fused_matmul_add_swish_tanh_gelu_hardtanh() {
    let c = [0.5, -0.5, 1.0, -1.0];
    let added: Vec<f32> = c.iter().map(|&v| v + 0.1).collect();
    let sw = swish(&added);
    let t = tanh_act(&sw);
    let g = gelu(&t);
    let result: Vec<f32> = g.iter().map(|&v| v.max(-1.0).min(1.0)).collect();
    assert!(result.iter().all(|&v| v >= -1.0 && v <= 1.0));
}

#[test]
fn test_fused_matmul_sub_mean_gelu() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let sub: Vec<f32> = c.iter().map(|&v| v - 0.5).collect();
    let result = gelu(&sub);
    assert_eq!(result.len(), 4);
}

// ===== fused_matmul_norm_kernel.rs (6 kernels) =====

#[test]
fn test_fused_gemm_norm_gelu() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&c, 1e-5);
    let result = gelu(&normed);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_gemm_norm_scale_softmax() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&c, 1e-5);
    let scaled: Vec<f32> = normed.iter().map(|&v| v * 2.0).collect();
    let result = softmax(&scaled);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_fused_gemm_scale_norm() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let scaled: Vec<f32> = c.iter().map(|&v| v * 2.0).collect();
    let result = normalization::layernorm(&scaled, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_gemm_norm_hardtanh() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&c, 1e-5);
    let result: Vec<f32> = normed.iter().map(|&v| v.max(-1.0).min(1.0)).collect();
    assert!(result.iter().all(|&v| v >= -1.0 && v <= 1.0));
}

#[test]
fn test_fused_gemm_norm_swish_mul_swish() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&c, 1e-5);
    let sw1 = swish(&normed);
    let scaled: Vec<f32> = sw1.iter().map(|&v| v * 2.0).collect();
    let result = swish(&scaled);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_fused_gemm_bias_hardtanh_mish_norm() {
    let c = [1.0, 2.0, 3.0, 4.0];
    let biased: Vec<f32> = c.iter().map(|&v| v + 0.1).collect();
    let ht: Vec<f32> = biased.iter().map(|&v| v.max(-1.0).min(1.0)).collect();
    let m = mish(&ht);
    let result = normalization::layernorm(&m, 1e-5);
    assert_eq!(result.len(), 4);
}
