//! CPU reference implementations of index/gather/scatter kernels.
//! Mirrors: index_ops_kernel.rs

/// Argmax: returns index of maximum value.
pub fn argmax(input: &[f32]) -> usize {
    let mut max_val = input[0];
    let mut max_idx = 0;
    for (i, &val) in input.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

/// Argmin: returns index of minimum value.
pub fn argmin(input: &[f32]) -> usize {
    let mut min_val = input[0];
    let mut min_idx = 0;
    for (i, &val) in input.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    min_idx
}

/// Gather: out[i] = input[index[i]]
pub fn gather(input: &[f32], index: &[usize], output: &mut [f32]) {
    for (i, &idx) in index.iter().enumerate() {
        output[i] = input[idx];
    }
}

/// Scatter: out[index[i]] = src[i]
pub fn scatter(src: &[f32], index: &[usize], output: &mut [f32]) {
    for (i, &idx) in index.iter().enumerate() {
        output[idx] = src[i];
    }
}

/// Scatter add: out[index[i]] += src[i]
pub fn scatter_add(src: &[f32], index: &[usize], output: &mut [f32]) {
    for (i, &idx) in index.iter().enumerate() {
        output[idx] += src[i];
    }
}

/// Index select: select rows by index.
pub fn index_select(input: &[f32], index: &[usize], output: &mut [f32], row_len: usize) {
    for (i, &idx) in index.iter().enumerate() {
        for j in 0..row_len {
            output[i * row_len + j] = input[idx * row_len + j];
        }
    }
}

/// Index copy: copy rows by index.
pub fn index_copy(src: &[f32], index: &[usize], output: &mut [f32], row_len: usize) {
    for (i, &idx) in index.iter().enumerate() {
        for j in 0..row_len {
            output[idx * row_len + j] = src[i * row_len + j];
        }
    }
}

/// Index add: add rows by index.
pub fn index_add(src: &[f32], index: &[usize], output: &mut [f32], row_len: usize) {
    for (i, &idx) in index.iter().enumerate() {
        for j in 0..row_len {
            output[idx * row_len + j] += src[i * row_len + j];
        }
    }
}

/// Embedding lookup: out[i] = weight[indices[i]]
pub fn embedding(weight: &[f32], indices: &[usize], output: &mut [f32], embed_dim: usize) {
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..embed_dim {
            output[i * embed_dim + j] = weight[idx * embed_dim + j];
        }
    }
}

/// Masked fill: out[i] = if mask[i] { fill_val } else { input[i] }
pub fn masked_fill(input: &[f32], mask: &[bool], output: &mut [f32], fill_val: f32) {
    for i in 0..input.len() {
        output[i] = if mask[i] { fill_val } else { input[i] };
    }
}

/// Inplace update: output[index[i]] = values[i]
pub fn inplace_update(values: &[f32], index: &[usize], output: &mut [f32]) {
    for (i, &idx) in index.iter().enumerate() {
        output[idx] = values[i];
    }
}

/// Take along dim (flat version).
pub fn take_along_dim(input: &[f32], index: &[usize], output: &mut [f32], inner: usize) {
    for i in 0..output.len() {
        let outer = i / inner;
        let idx = index[i];
        let src_pos = outer * inner + idx;
        output[i] = input[src_pos];
    }
}
