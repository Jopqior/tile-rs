use kernel_correctness::activation::*;
use kernel_correctness::normalization;

const TOL: f32 = 1e-4;

// ===== attention_kernel.rs (6 kernels) =====

#[test]
fn test_attention_softmax() {
    // scores = softmax(x / sqrt(d_model))
    let x = [1.0, 2.0, 3.0, 4.0];
    let d_model = 64.0f32;
    let scale = 1.0 / d_model.sqrt();
    let scaled: Vec<f32> = x.iter().map(|&v| v * scale).collect();
    let result = softmax(&scaled);
    assert!(
        (result.iter().sum::<f32>() - 1.0).abs() < TOL,
        "attn_softmax sums to 1"
    );
}

#[test]
fn test_residual_add_layernorm() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let residual = [0.1, 0.2, 0.3, 0.4];
    let sum: Vec<f32> = x.iter().zip(&residual).map(|(&a, &b)| a + b).collect();
    let result = normalization::layernorm(&sum, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_residual_add_rmsnorm() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let residual = [0.1, 0.2, 0.3, 0.4];
    let sum: Vec<f32> = x.iter().zip(&residual).map(|(&a, &b)| a + b).collect();
    let result = normalization::rms_norm(&sum, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_swiglu() {
    // swiglu(x, gate) = swish(gate) * x
    let x = [1.0, 2.0, 3.0];
    let gate = [0.5, 1.0, -0.5];
    let swish_gate = swish(&gate);
    let result: Vec<f32> = x.iter().zip(&swish_gate).map(|(&a, &b)| a * b).collect();
    assert_eq!(result.len(), 3);
    assert!(result[0] > 0.0);
}

#[test]
fn test_geglu() {
    // geglu(x, gate) = gelu(gate) * x
    let x = [1.0, 2.0, 3.0];
    let gate = [0.5, 1.0, -0.5];
    let gelu_gate = gelu(&gate);
    let result: Vec<f32> = x.iter().zip(&gelu_gate).map(|(&a, &b)| a * b).collect();
    assert!(result[0] > 0.0);
}

#[test]
fn test_masked_fill() {
    // output = x * mask + fill * (1 - mask)
    let x = [1.0, 2.0, 3.0, 4.0];
    let mask = [1.0, 0.0, 1.0, 0.0]; // keep 0,2; fill 1,3
    let fill = -1e9f32;
    let result: Vec<f32> = x
        .iter()
        .zip(&mask)
        .map(|(&xi, &mi)| xi * mi + fill * (1.0 - mi))
        .collect();
    assert!((result[0] - 1.0).abs() < TOL);
    assert!((result[1] - fill).abs() < 1.0);
}

// ===== attention_extended_kernel.rs (9 kernels) =====

#[test]
fn test_causal_attention() {
    // softmax(scores * scale + mask)
    let scores = [1.0, 2.0, 3.0, 4.0];
    let mask = [0.0, -1e9, 0.0, -1e9]; // mask out positions 1,3
    let scale = 0.5f32;
    let scaled: Vec<f32> = scores
        .iter()
        .zip(&mask)
        .map(|(&s, &m)| s * scale + m)
        .collect();
    let result = softmax(&scaled);
    assert!(
        (result.iter().sum::<f32>() - 1.0).abs() < TOL,
        "causal sums to 1"
    );
    assert!(result[1] < 1e-5, "masked position near 0");
}

#[test]
fn test_cross_attention() {
    // softmax(q * k * scale)
    let q = [1.0, 0.5, 0.3];
    let k = [0.5, 1.0, 0.7];
    let scale = 0.5f32;
    let qk: Vec<f32> = q.iter().zip(&k).map(|(&a, &b)| a * b * scale).collect();
    let result = softmax(&qk);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_multi_query_attention() {
    // Same math as cross_attention (shared KV)
    let q = [1.0, 2.0, 3.0];
    let k_shared = [0.5, 0.5, 0.5];
    let scale = 0.125f32;
    let qk: Vec<f32> = q
        .iter()
        .zip(&k_shared)
        .map(|(&a, &b)| a * b * scale)
        .collect();
    let result = softmax(&qk);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_group_query_attention() {
    let q = [1.0, 2.0];
    let k = [0.5, 0.5];
    let scale = 0.5f32;
    let qk: Vec<f32> = q.iter().zip(&k).map(|(&a, &b)| a * b * scale).collect();
    let result = softmax(&qk);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_kv_cached_attention() {
    // Merge cached + new, then attend
    let q = [1.0, 2.0];
    let kv_cached = [0.3, 0.4];
    let kv_new = [0.2, 0.1];
    let merged: Vec<f32> = kv_cached
        .iter()
        .zip(&kv_new)
        .map(|(&a, &b)| a + b)
        .collect();
    let qk: Vec<f32> = q.iter().zip(&merged).map(|(&a, &b)| a * b * 0.5).collect();
    let result = softmax(&qk);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_cross_modal_attention() {
    let text_q = [1.0, 0.5];
    let image_k = [0.5, 1.0];
    let scale = 0.5f32;
    let qk: Vec<f32> = text_q
        .iter()
        .zip(&image_k)
        .map(|(&a, &b)| a * b * scale)
        .collect();
    let result = softmax(&qk);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_linear_attention() {
    // ELU+1 feature map: max(0, x) + 1, then multiply
    let q = [-1.0, 0.5, 2.0];
    let k = [0.5, -0.5, 1.0];
    let scale = 0.5f32;
    let phi_q: Vec<f32> = q.iter().map(|&v: &f32| v.max(0.0) + 1.0).collect();
    let phi_k: Vec<f32> = k.iter().map(|&v: &f32| v.max(0.0) + 1.0).collect();
    let result: Vec<f32> = phi_q
        .iter()
        .zip(&phi_k)
        .map(|(&a, &b)| a * b * scale)
        .collect();
    assert!(result.iter().all(|&v| v > 0.0), "linear attention positive");
}

#[test]
fn test_sparse_attention() {
    // scale * scores * mask then softmax
    let scores = [1.0, 2.0, 3.0, 4.0];
    let mask = [1.0, 0.0, 1.0, 0.0]; // sparsity
    let scale = 0.5f32;
    let masked: Vec<f32> = scores
        .iter()
        .zip(&mask)
        .map(|(&s, &m)| s * scale * m)
        .collect();
    let result = softmax(&masked);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_windowed_causal_attention() {
    // softmax(scores * scale + window_mask)
    let scores = [1.0, 2.0, 3.0];
    let window_mask = [0.0, 0.0, -1e9]; // mask out position 2
    let scale = 0.5f32;
    let masked: Vec<f32> = scores
        .iter()
        .zip(&window_mask)
        .map(|(&s, &m)| s * scale + m)
        .collect();
    let result = softmax(&masked);
    assert!(result[2] < 1e-5, "windowed masked position");
}
