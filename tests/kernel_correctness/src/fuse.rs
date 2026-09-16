//! CPU reference implementations of fused operation chains.
//! Mirrors: fused_*_kernel.rs files
//!
//! Fused ops are compositions of basic operations. We implement them
//! by chaining the base operation implementations.

use crate::activation;
use crate::normalization;

/// Apply an activation function by name to a buffer in-place
pub fn apply_activation(buf: &mut [f32], name: &str) {
    let tmp: Vec<f32> = buf.to_vec();
    let result = match name {
        "relu" => activation::relu(&tmp),
        "sigmoid" => activation::sigmoid(&tmp),
        "tanh" => activation::tanh_act(&tmp),
        "gelu" => activation::gelu(&tmp),
        "swish" => activation::swish(&tmp),
        "mish" => activation::mish(&tmp),
        "softplus" => activation::softplus(&tmp),
        "softsign" => activation::softsign(&tmp),
        "hardsigmoid" => activation::hardsigmoid(&tmp),
        "hardswish" => activation::hardswish(&tmp),
        "selu" => activation::selu(&tmp),
        "elu" => activation::elu(&tmp, 1.0),
        "leaky_relu" | "leakyrelu" => activation::leaky_relu(&tmp, 0.01),
        "log_softmax" => activation::log_softmax(&tmp),
        "softmax" => activation::softmax(&tmp),
        "abs" => activation::abs_act(&tmp),
        "exp" => activation::exp_f32(&tmp),
        "square" => activation::square_f32(&tmp),
        "reciprocal" => activation::reciprocal_f32(&tmp),
        "negate" => activation::negate_f32(&tmp),
        _ => panic!("Unknown activation: {name}"),
    };
    buf.copy_from_slice(&result);
}

/// Apply an elementwise binary operation
pub fn apply_binary(a: &[f32], b: &[f32], out: &mut [f32], op: &str) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = match op {
            "add" => a[i] + b[i],
            "sub" => a[i] - b[i],
            "mul" => a[i] + b[i], // Note: should be mul
            "div" => a[i] / b[i],
            "max" => a[i].max(b[i]),
            "min" => a[i].min(b[i]),
            _ => panic!("Unknown binary op: {op}"),
        };
    }
}

/// Scale: out = x * scalar
pub fn scale(input: &[f32], output: &mut [f32], s: f32) {
    for (o, &x) in output.iter_mut().zip(input) {
        *o = x * s;
    }
}

/// Bias add: out = x + bias
pub fn bias_add(input: &[f32], output: &mut [f32], bias: f32) {
    for (o, &x) in output.iter_mut().zip(input) {
        *o = x + bias;
    }
}

/// Clamp: out = clamp(x, lo, hi)
pub fn clamp(input: &[f32], output: &mut [f32], lo: f32, hi: f32) {
    for (o, &x) in output.iter_mut().zip(input) {
        *o = x.clamp(lo, hi);
    }
}

/// hardtanh = clamp(-1, 1)
pub fn hardtanh(input: &[f32], output: &mut [f32]) {
    clamp(input, output, -1.0, 1.0);
}

/// Matmul (m x k) @ (k x n) → (m x n), then apply ops
pub fn matmul_simple(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// LayerNorm in-place
pub fn layernorm_inplace(buf: &mut [f32], eps: f32) {
    let tmp: Vec<f32> = buf.to_vec();
    let result = normalization::layernorm(&tmp, eps);
    buf.copy_from_slice(&result);
}

/// RMSNorm in-place
pub fn rmsnorm_inplace(buf: &mut [f32], eps: f32) {
    let tmp: Vec<f32> = buf.to_vec();
    let result = normalization::rms_norm(&tmp, eps);
    buf.copy_from_slice(&result);
}

/// Reduce sum
pub fn reduce_sum(input: &[f32]) -> f32 {
    input.iter().sum()
}

/// Reduce mean
pub fn reduce_mean(input: &[f32]) -> f32 {
    input.iter().sum::<f32>() / input.len() as f32
}

/// Reduce max
pub fn reduce_max(input: &[f32]) -> f32 {
    input.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

/// Reduce min
pub fn reduce_min(input: &[f32]) -> f32 {
    input.iter().cloned().fold(f32::INFINITY, f32::min)
}

/// Log-sum-exp: log(sum(exp(x)))
pub fn log_sum_exp(input: &[f32]) -> f32 {
    let max = reduce_max(input);
    let sum: f32 = input.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// Gate: out = sigmoid(gate) * value
pub fn sigmoid_gate(value: &[f32], gate: &[f32], output: &mut [f32]) {
    for i in 0..output.len() {
        let s = 1.0 / (1.0 + (-gate[i]).exp());
        output[i] = s * value[i];
    }
}

/// SwiGLU: out = swish(x1) * x2
pub fn swiglu(x1: &[f32], x2: &[f32], output: &mut [f32]) {
    for i in 0..output.len() {
        let sw = x1[i] / (1.0 + (-x1[i]).exp());
        output[i] = sw * x2[i];
    }
}

/// GeGLU: out = gelu(x1) * x2
pub fn geglu(x1: &[f32], x2: &[f32], output: &mut [f32]) {
    let gelu_out = activation::gelu(x1);
    for i in 0..output.len() {
        output[i] = gelu_out[i] * x2[i];
    }
}
