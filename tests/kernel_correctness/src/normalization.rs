//! CPU reference implementations for normalization operations.

pub fn layernorm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter().map(|&v| (v - mean) * inv_std).collect()
}

pub fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n + eps).sqrt();
    x.iter().map(|&v| v / rms).collect()
}

pub fn l1_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v.abs()).sum()
}

pub fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

pub fn l2_normalize(x: &[f32], eps: f32) -> Vec<f32> {
    let norm = l2_norm(x) + eps;
    x.iter().map(|&v| v / norm).collect()
}

pub fn frobenius_norm(x: &[f32]) -> f32 {
    l2_norm(x)
}

/// Batch norm (element-wise, pre-computed mean/var as vectors)
pub fn batch_norm_vec(x: &[f32], mean: &[f32], var: &[f32], eps: f32) -> Vec<f32> {
    x.iter()
        .zip(mean.iter().zip(var.iter()))
        .map(|(&xi, (&mi, &vi))| (xi - mi) / (vi + eps).sqrt())
        .collect()
}

/// Batch norm (scalar mean/var)
pub fn batch_norm(x: &[f32], mean: f32, var: f32, gamma: f32, beta: f32, eps: f32) -> Vec<f32> {
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .map(|&v| (v - mean) * inv_std * gamma + beta)
        .collect()
}

/// Group norm (simplified as layernorm per group)
pub fn group_norm(x: &[f32], _num_groups: usize, eps: f32) -> Vec<f32> {
    layernorm(x, eps)
}

/// Instance norm (same as layernorm for 1D)
pub fn instance_norm(x: &[f32], eps: f32) -> Vec<f32> {
    layernorm(x, eps)
}
