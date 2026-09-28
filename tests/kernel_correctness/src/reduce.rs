//! CPU reference implementations for reduction operations.

pub fn reduce_max(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

pub fn reduce_min(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::INFINITY, f32::min)
}

pub fn reduce_sum(x: &[f32]) -> f32 {
    x.iter().sum()
}

pub fn reduce_mean(x: &[f32]) -> f32 {
    x.iter().sum::<f32>() / x.len() as f32
}

/// Product reduction (using exp(sum(log(x))) for positive inputs)
pub fn reduce_prod(x: &[f32]) -> f32 {
    x.iter().product()
}

pub fn cumsum(x: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(x.len());
    let mut acc = 0.0f32;
    for &v in x {
        acc += v;
        result.push(acc);
    }
    result
}

pub fn cumsum_exclusive(x: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(x.len());
    let mut acc = 0.0f32;
    for &v in x {
        result.push(acc);
        acc += v;
    }
    result
}

pub fn cumsum_reverse(x: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; x.len()];
    let mut acc = 0.0f32;
    for i in (0..x.len()).rev() {
        acc += x[i];
        result[i] = acc;
    }
    result
}

pub fn cumprod(x: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(x.len());
    let mut acc = 1.0f32;
    for &v in x {
        acc *= v;
        result.push(acc);
    }
    result
}
