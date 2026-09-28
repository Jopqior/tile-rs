//! CPU reference implementations for activation functions.
//! These match the math in compiletest kernels (relu, sigmoid, etc.)

use std::f32::consts::PI;

pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

pub fn sigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

pub fn tanh_act(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.tanh()).collect()
}

/// GELU (erf approximation): x * 0.5 * (1 + erf(x / sqrt(2)))
pub fn gelu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let cdf = 0.5 * (1.0 + erf(v / std::f32::consts::SQRT_2));
            v * cdf
        })
        .collect()
}

/// GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_tanh(x: &[f32]) -> Vec<f32> {
    let c = (2.0 / PI).sqrt();
    x.iter()
        .map(|&v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh()))
        .collect()
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&v| v / sum).collect()
}

pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + x.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
    x.iter().map(|&v| v - lse).collect()
}

pub fn elu(x: &[f32], alpha: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
        .collect()
}

pub fn selu(x: &[f32]) -> Vec<f32> {
    let alpha = 1.6732632;
    let lambda = 1.0507010;
    x.iter()
        .map(|&v| lambda * if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
        .collect()
}

pub fn swish(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

pub fn mish(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let sp = (1.0 + v.exp()).ln();
            v * sp.tanh()
        })
        .collect()
}

pub fn softplus(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| (1.0 + v.exp()).ln()).collect()
}

pub fn softsign(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + v.abs())).collect()
}

pub fn hardsigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| (v / 6.0 + 0.5).clamp(0.0, 1.0)).collect()
}

pub fn hardswish(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| v * (v / 6.0 + 0.5).clamp(0.0, 1.0))
        .collect()
}

pub fn leaky_relu(x: &[f32], alpha: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| if v > 0.0 { v } else { alpha * v })
        .collect()
}

pub fn abs_act(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.abs()).collect()
}

pub fn scalar_mul(x: &[f32], s: f32) -> Vec<f32> {
    x.iter().map(|&v| v * s).collect()
}

pub fn hardtanh(x: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
    x.iter().map(|&v| v.clamp(min_val, max_val)).collect()
}

pub fn relu6(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0).min(6.0)).collect()
}

// f32 unary ops
pub fn exp_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.exp()).collect()
}
pub fn ln_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.ln()).collect()
}
pub fn sqrt_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.sqrt()).collect()
}
pub fn rsqrt_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / v.sqrt()).collect()
}
pub fn reciprocal_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / v).collect()
}
pub fn negate_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| -v).collect()
}
pub fn square_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * v).collect()
}
pub fn cube_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * v * v).collect()
}

// Elementwise binary ops
pub fn add(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(&a, &b)| a + b).collect()
}
pub fn sub(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(&a, &b)| a - b).collect()
}
pub fn mul(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(&a, &b)| a * b).collect()
}
pub fn div(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(&a, &b)| a / b).collect()
}
pub fn max_ew(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(&a, &b)| a.max(b)).collect()
}
pub fn min_ew(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(&a, &b)| a.min(b)).collect()
}

/// Abramowitz-Stegun erf approximation
fn erf(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-a * a).exp();
    sign * y
}
