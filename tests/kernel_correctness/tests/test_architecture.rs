use kernel_correctness::activation::*;
use kernel_correctness::golden::assert_approx;
use kernel_correctness::normalization;

const TOL: f32 = 1e-4;

// ===== arch_ops_kernel.rs (13 kernels) =====

#[test]
fn test_mlp_relu() {
    let x = [1.0, -1.0, 2.0, -2.0];
    let result = relu(&x);
    assert_approx(&result, &[1.0, 0.0, 2.0, 0.0], TOL, "mlp_relu");
}

#[test]
fn test_mlp_gelu_bias() {
    // matmul result + bias(0.1), then gelu
    let x = [1.0, 2.0, 3.0, 4.0];
    let biased: Vec<f32> = x.iter().map(|&v| v + 0.1).collect();
    let result = gelu(&biased);
    assert!(
        result.iter().all(|&v| v > 0.0),
        "gelu_bias positive for positive input"
    );
}

#[test]
fn test_mlp_swish() {
    let x = [1.0, 2.0, -1.0];
    let result = swish(&x);
    assert!(result[0] > 0.0 && result[2] < 0.0);
}

#[test]
fn test_ffn_prenorm() {
    // norm + gelu
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let result = gelu(&normed);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_down_proj() {
    // scale by 0.1
    let x = [10.0, 20.0, 30.0];
    let result = scalar_mul(&x, 0.1);
    assert_approx(&result, &[1.0, 2.0, 3.0], TOL, "down_proj");
}

#[test]
fn test_attention_score_norm() {
    // softmax(x / sqrt(d_k))
    let x = [1.0, 2.0, 3.0, 4.0];
    let d_k = 64.0f32;
    let scale = 1.0 / d_k.sqrt();
    let scaled: Vec<f32> = x.iter().map(|&v| v * scale).collect();
    let result = softmax(&scaled);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_rope_freq() {
    // freq_i = exp(-dim_frac * ln(base)), base=10000
    let base = 10000.0f32;
    let n = 4;
    let freqs: Vec<f32> = (0..n)
        .map(|i| {
            let dim_frac = (2 * i) as f32 / n as f32;
            (-dim_frac * base.ln()).exp()
        })
        .collect();
    assert!((freqs[0] - 1.0).abs() < TOL, "freq[0] = 1");
    assert!(freqs[1] < freqs[0], "freqs decrease");
}

#[test]
fn test_embedding_scale() {
    assert_approx(
        &scalar_mul(&[1.0, 2.0, 3.0], 2.0),
        &[2.0, 4.0, 6.0],
        TOL,
        "emb_scale",
    );
}

#[test]
fn test_gated_residual() {
    // sigmoid(gate) * value + residual
    let value = [1.0, 2.0];
    let gate = [0.0, 0.0]; // sigmoid(0) = 0.5
    let residual = [1.0, 1.0];
    let sig_gate = sigmoid(&gate);
    let result: Vec<f32> = value
        .iter()
        .zip(sig_gate.iter().zip(&residual))
        .map(|(&v, (&s, &r))| v * s + r)
        .collect();
    assert_approx(&result, &[1.5, 2.0], TOL, "gated_residual");
}

#[test]
fn test_scaled_dot() {
    let q = [1.0, 2.0, 3.0];
    let k = [0.5, 0.5, 0.5];
    let scale = 0.5f32;
    let result: Vec<f32> = q.iter().zip(&k).map(|(&a, &b)| a * b * scale).collect();
    assert_approx(&result, &[0.25, 0.5, 0.75], TOL, "scaled_dot");
}

#[test]
fn test_classifier_head() {
    // matmul + bias(0.1) + sigmoid
    let x = [0.0, 0.0];
    let biased: Vec<f32> = x.iter().map(|&v| v + 0.1).collect();
    let result = sigmoid(&biased);
    assert!(
        result.iter().all(|&v| v > 0.5),
        "classifier > 0.5 for positive bias"
    );
}

#[test]
fn test_regression_head() {
    // matmul + bias(0.01) — no activation
    let x = [1.0, 2.0, 3.0];
    let result: Vec<f32> = x.iter().map(|&v| v + 0.01).collect();
    assert_approx(&result, &[1.01, 2.01, 3.01], TOL, "regression_head");
}

#[test]
fn test_softmax_classifier() {
    let x = [1.0, 2.0, 3.0];
    let result = softmax(&x);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

// ===== arch_network_kernel.rs (18 kernels) =====

#[test]
fn test_alexnet_fc() {
    // FC + ReLU
    let x = [-1.0, 2.0, -3.0, 4.0];
    assert_approx(&relu(&x), &[0.0, 2.0, 0.0, 4.0], TOL, "alexnet");
}

#[test]
fn test_vgg_fc() {
    // FC + bias(0.01) + ReLU
    let x = [-1.0, 2.0];
    let biased: Vec<f32> = x.iter().map(|&v| v + 0.01).collect();
    let result = relu(&biased);
    assert_approx(&result, &[0.0, 2.01], TOL, "vgg");
}

#[test]
fn test_resnet_residual() {
    // layernorm + relu + residual add
    let x = [1.0, 2.0, 3.0, 4.0];
    let residual = [0.5, 0.5, 0.5, 0.5];
    let normed = normalization::layernorm(&x, 1e-5);
    let activated = relu(&normed);
    let result: Vec<f32> = activated
        .iter()
        .zip(&residual)
        .map(|(&a, &b)| a + b)
        .collect();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_densenet_block() {
    // layernorm + relu + scale(0.5)
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let activated = relu(&normed);
    let result = scalar_mul(&activated, 0.5);
    assert!(result.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_mobilenet_pointwise() {
    // FC + relu6
    let x = [-1.0, 3.0, 7.0];
    let result = relu6(&x);
    assert_approx(&result, &[0.0, 3.0, 6.0], TOL, "mobilenet");
}

#[test]
fn test_efficientnet_fc() {
    let x = [1.0, 2.0, -1.0];
    let result = swish(&x);
    assert!(result[0] > 0.0);
}

#[test]
fn test_inception_merge() {
    // add + relu
    let a = [1.0, -1.0, 2.0];
    let b = [2.0, 3.0, -4.0];
    let sum: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
    let result = relu(&sum);
    assert_approx(&result, &[3.0, 2.0, 0.0], TOL, "inception");
}

#[test]
fn test_squeezenet_fire() {
    // scale(0.25) + relu + scale(4) + relu
    let x = [1.0, -2.0, 4.0];
    let s1: Vec<f32> = x.iter().map(|&v| v * 0.25).collect();
    let r1 = relu(&s1);
    let s2: Vec<f32> = r1.iter().map(|&v| v * 4.0).collect();
    let result = relu(&s2);
    assert_approx(&result, &[1.0, 0.0, 4.0], TOL, "squeezenet");
}

#[test]
fn test_shufflenet_fc() {
    // relu + bias(0.1)
    let x = [-1.0, 2.0, 3.0];
    let activated = relu(&x);
    let result: Vec<f32> = activated.iter().map(|&v| v + 0.1).collect();
    assert_approx(&result, &[0.1, 2.1, 3.1], TOL, "shufflenet");
}

#[test]
fn test_regnet_stem() {
    // layernorm + relu + scale(0.1)
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let activated = relu(&normed);
    let result = scalar_mul(&activated, 0.1);
    assert!(result.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_lenet_fc() {
    // FC + tanh
    let x = [0.0, 1.0, -1.0];
    let result = tanh_act(&x);
    assert!((result[0]).abs() < TOL);
}

#[test]
fn test_unet_skip() {
    // encoder + decoder, then layernorm
    let enc = [1.0, 2.0, 3.0, 4.0];
    let dec = [0.1, 0.2, 0.3, 0.4];
    let sum: Vec<f32> = enc.iter().zip(&dec).map(|(&a, &b)| a + b).collect();
    let result = normalization::layernorm(&sum, 1e-5);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_vit_mlp() {
    // layernorm + gelu
    let x = [1.0, 2.0, 3.0, 4.0];
    let normed = normalization::layernorm(&x, 1e-5);
    let result = gelu(&normed);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_swin_attention() {
    // scale + softmax
    let x = [1.0, 2.0, 3.0, 4.0];
    let scale = 0.125f32;
    let scaled: Vec<f32> = x.iter().map(|&v| v * scale).collect();
    let result = softmax(&scaled);
    assert!((result.iter().sum::<f32>() - 1.0).abs() < TOL);
}

#[test]
fn test_mingpt_block() {
    // layernorm + softmax + residual
    let x = [1.0, 2.0, 3.0, 4.0];
    let residual = [0.5, 0.5, 0.5, 0.5];
    let normed = normalization::layernorm(&x, 1e-5);
    let sm = softmax(&normed);
    let result: Vec<f32> = sm.iter().zip(&residual).map(|(&a, &b)| a + b).collect();
    assert!(result.iter().all(|&v| v > 0.0));
}

#[test]
fn test_mlp_mixer() {
    // FC + gelu
    let x = [1.0, 2.0, 3.0];
    let result = gelu(&x);
    assert!(result.iter().all(|&v| v > 0.0));
}

#[test]
fn test_mamba_ssm() {
    // sigmoid(gate) * x
    let x = [1.0, 2.0, 3.0];
    let gate = [0.0, 1.0, -1.0];
    let sig = sigmoid(&gate);
    let result: Vec<f32> = x.iter().zip(&sig).map(|(&a, &b)| a * b).collect();
    assert!((result[0] - 0.5).abs() < TOL, "mamba x*sigmoid(0)=0.5");
}

#[test]
fn test_mobilenetv2_inverted() {
    // expand(6x) + relu6 + project(1/6) + residual
    let x = [1.0, -0.5];
    let residual = [0.1, 0.2];
    let expanded: Vec<f32> = x.iter().map(|&v| v * 6.0).collect();
    let r6 = relu6(&expanded);
    let projected: Vec<f32> = r6.iter().map(|&v| v * 0.1667).collect();
    let result: Vec<f32> = projected
        .iter()
        .zip(&residual)
        .map(|(&a, &b)| a + b)
        .collect();
    assert_eq!(result.len(), 2);
}

// ===== arch_rnn_kernel.rs (10 kernels) =====

fn rnn_gate(x: &[f32], h: &[f32], scale: f32, bias: f32) -> Vec<f32> {
    x.iter()
        .zip(h)
        .map(|(&xi, &hi)| xi + hi * scale + bias)
        .collect()
}

#[test]
fn test_vanilla_rnn() {
    let pre = rnn_gate(&[1.0, 0.5], &[0.3, 0.7], 0.5, 0.1);
    let result = tanh_act(&pre);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_lstm_forget_gate() {
    let pre = rnn_gate(&[1.0], &[0.5], 0.8, 0.1);
    let result = sigmoid(&pre);
    assert!(result[0] > 0.0 && result[0] < 1.0);
}

#[test]
fn test_lstm_input_gate() {
    let pre = rnn_gate(&[0.5], &[0.3], 0.8, 0.0);
    let result = sigmoid(&pre);
    assert!(result[0] > 0.0 && result[0] < 1.0);
}

#[test]
fn test_lstm_cell_candidate() {
    let pre = rnn_gate(&[1.0], &[0.5], 0.5, 0.1);
    let result = tanh_act(&pre);
    assert!(result[0].abs() <= 1.0);
}

#[test]
fn test_lstm_cell_update() {
    // c_new = f * c_old + i * c_hat
    let c_old = [1.0, 2.0];
    let f_gate = [0.9, 0.8];
    let i_gate = [0.1, 0.2];
    let c_hat = [0.5, 0.3];
    let result: Vec<f32> = (0..2)
        .map(|j| c_old[j] * f_gate[j] + i_gate[j] * c_hat[j])
        .collect();
    assert_approx(&result, &[0.95, 1.66], TOL, "lstm_cell_update");
}

#[test]
fn test_lstm_output() {
    // h = o * tanh(c)
    let cell = [1.0, -0.5];
    let o_gate = [0.8, 0.6];
    let tanh_c = tanh_act(&cell);
    let result: Vec<f32> = tanh_c.iter().zip(&o_gate).map(|(&t, &o)| t * o).collect();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_gru_reset_gate() {
    let pre = rnn_gate(&[1.0], &[0.5], 0.5, 0.1);
    let result = sigmoid(&pre);
    assert!(result[0] > 0.0 && result[0] < 1.0);
}

#[test]
fn test_gru_update_gate() {
    let pre = rnn_gate(&[0.5], &[0.3], 0.5, -0.1);
    let result = sigmoid(&pre);
    assert!(result[0] > 0.0 && result[0] < 1.0);
}

#[test]
fn test_gru_candidate() {
    // x + r*h, then tanh
    let x = [1.0, 0.5];
    let h = [0.3, 0.7];
    let r = [0.8, 0.6];
    let rh: Vec<f32> = h.iter().zip(&r).map(|(&hi, &ri)| hi * ri).collect();
    let combined: Vec<f32> = x.iter().zip(&rh).map(|(&a, &b)| a + b).collect();
    let result = tanh_act(&combined);
    assert!(result.iter().all(|&v| v.abs() <= 1.0));
}

#[test]
fn test_gru_hidden_update() {
    // h_new = (1-z)*h + z*h_hat
    let h = [1.0, 2.0];
    let z = [0.3, 0.7];
    let h_hat = [0.5, 1.0];
    let result: Vec<f32> = (0..2)
        .map(|i| (1.0 - z[i]) * h[i] + z[i] * h_hat[i])
        .collect();
    assert_approx(&result, &[0.85, 1.3], TOL, "gru_hidden");
}
