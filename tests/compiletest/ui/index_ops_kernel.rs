// build-pass

// Index/gather/scatter operation kernels.
// Maps to MultiKernelBench/reference/index/ category.
// All use scalar loops with indirect pointer access on GM pointers.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Argmax over a dimension: returns index of maximum value
/// Maps to index/argmax_over_a_dimension.py
#[tile_std::tile_kernel]
pub fn argmax(input: *const f32, output: *mut u32, len: *const u32) {
    unsafe {
        let n = *len;
        if n == 0 { return; }
        let mut max_val = *input;
        let mut max_idx = 0u32;
        let mut i = 1u32;
        loop {
            if i >= n { break; }
            let val = *input.wrapping_add(i as usize);
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
            i = i + 1;
        }
        *output = max_idx;
    }
}

/// Argmin over a dimension: returns index of minimum value
/// Maps to index/argmin_over_a_dimension.py
#[tile_std::tile_kernel]
pub fn argmin(input: *const f32, output: *mut u32, len: *const u32) {
    unsafe {
        let n = *len;
        if n == 0 { return; }
        let mut min_val = *input;
        let mut min_idx = 0u32;
        let mut i = 1u32;
        loop {
            if i >= n { break; }
            let val = *input.wrapping_add(i as usize);
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
            i = i + 1;
        }
        *output = min_idx;
    }
}

/// Gather: out[i] = input[index[i]]
/// Maps to index/gather.py
#[tile_std::tile_kernel]
pub fn gather(
    input: *const f32, index: *const u32, output: *mut f32, len: *const u32,
) {
    unsafe {
        let n = *len;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let idx = *index.wrapping_add(i as usize);
            *output.wrapping_add(i as usize) = *input.wrapping_add(idx as usize);
            i = i + 1;
        }
    }
}

/// Scatter: out[index[i]] = src[i]
/// Maps to index/scatter.py
#[tile_std::tile_kernel]
pub fn scatter(
    src: *const f32, index: *const u32, output: *mut f32, len: *const u32,
) {
    unsafe {
        let n = *len;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let idx = *index.wrapping_add(i as usize);
            *output.wrapping_add(idx as usize) = *src.wrapping_add(i as usize);
            i = i + 1;
        }
    }
}

/// Scatter add: out[index[i]] += src[i]
/// Maps to index/scatter_add.py
#[tile_std::tile_kernel]
pub fn scatter_add(
    src: *const f32, index: *const u32, output: *mut f32, len: *const u32,
) {
    unsafe {
        let n = *len;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let idx = *index.wrapping_add(i as usize);
            let cur = *output.wrapping_add(idx as usize);
            *output.wrapping_add(idx as usize) = cur + *src.wrapping_add(i as usize);
            i = i + 1;
        }
    }
}

/// Index select: select rows by index. out[i] = input[index[i] * row_len .. (index[i]+1) * row_len]
/// Maps to index/index_select.py
#[tile_std::tile_kernel]
pub fn index_select(
    input: *const f32, index: *const u32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let num_idx = *params;
        let row_len = *params.wrapping_add(1);
        let mut i = 0u32;
        loop {
            if i >= num_idx { break; }
            let idx = *index.wrapping_add(i as usize);
            let mut j = 0u32;
            loop {
                if j >= row_len { break; }
                let src_pos = (idx * row_len + j) as usize;
                let dst_pos = (i * row_len + j) as usize;
                *output.wrapping_add(dst_pos) = *input.wrapping_add(src_pos);
                j = j + 1;
            }
            i = i + 1;
        }
    }
}

/// Index copy: copy rows by index. output[index[i]] = src[i] (row-level)
/// Maps to index/index_copy.py
#[tile_std::tile_kernel]
pub fn index_copy(
    src: *const f32, index: *const u32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let num_idx = *params;
        let row_len = *params.wrapping_add(1);
        let mut i = 0u32;
        loop {
            if i >= num_idx { break; }
            let idx = *index.wrapping_add(i as usize);
            let mut j = 0u32;
            loop {
                if j >= row_len { break; }
                let src_pos = (i * row_len + j) as usize;
                let dst_pos = (idx * row_len + j) as usize;
                *output.wrapping_add(dst_pos) = *src.wrapping_add(src_pos);
                j = j + 1;
            }
            i = i + 1;
        }
    }
}

/// Index add: add rows by index. output[index[i]] += src[i] (row-level)
/// Maps to index/index_add.py
#[tile_std::tile_kernel]
pub fn index_add(
    src: *const f32, index: *const u32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let num_idx = *params;
        let row_len = *params.wrapping_add(1);
        let mut i = 0u32;
        loop {
            if i >= num_idx { break; }
            let idx = *index.wrapping_add(i as usize);
            let mut j = 0u32;
            loop {
                if j >= row_len { break; }
                let src_pos = (i * row_len + j) as usize;
                let dst_pos = (idx * row_len + j) as usize;
                let cur = *output.wrapping_add(dst_pos);
                *output.wrapping_add(dst_pos) = cur + *src.wrapping_add(src_pos);
                j = j + 1;
            }
            i = i + 1;
        }
    }
}

/// Embedding lookup: out[i] = weight[indices[i]] (table lookup)
/// Maps to index/embedding.py
#[tile_std::tile_kernel]
pub fn embedding(
    weight: *const f32, indices: *const u32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let num_idx = *params;
        let embed_dim = *params.wrapping_add(1);
        let mut i = 0u32;
        loop {
            if i >= num_idx { break; }
            let idx = *indices.wrapping_add(i as usize);
            let mut j = 0u32;
            loop {
                if j >= embed_dim { break; }
                let src_pos = (idx * embed_dim + j) as usize;
                let dst_pos = (i * embed_dim + j) as usize;
                *output.wrapping_add(dst_pos) = *weight.wrapping_add(src_pos);
                j = j + 1;
            }
            i = i + 1;
        }
    }
}

/// Masked fill: out[i] = mask[i] != 0 ? fill_val : input[i]
/// Maps to index/masked_fill.py
#[tile_std::tile_kernel]
pub fn masked_fill(
    input: *const f32, mask: *const u32, output: *mut f32, params: *const f32,
) {
    unsafe {
        let fill_val = *params;
        let n_ptr = params.wrapping_add(1) as *const u32;
        let n = *n_ptr;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let m = *mask.wrapping_add(i as usize);
            if m != 0 {
                *output.wrapping_add(i as usize) = fill_val;
            } else {
                *output.wrapping_add(i as usize) = *input.wrapping_add(i as usize);
            }
            i = i + 1;
        }
    }
}

/// Inplace update: write values at specific indices. output[index[i]] = values[i]
/// Maps to index/inplace_update.py
#[tile_std::tile_kernel]
pub fn inplace_update(
    values: *const f32, index: *const u32, output: *mut f32, len: *const u32,
) {
    unsafe {
        let n = *len;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let idx = *index.wrapping_add(i as usize);
            *output.wrapping_add(idx as usize) = *values.wrapping_add(i as usize);
            i = i + 1;
        }
    }
}

/// Take along dim: out[i] = input[index[i]] along a dimension (flat version)
/// Maps to index/take_along_dim.py
#[tile_std::tile_kernel]
pub fn take_along_dim(
    input: *const f32, index: *const u32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let n = *params; // number of output elements
        let inner = *params.wrapping_add(1); // inner dimension size
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let outer = i / inner;
            let j = i - outer * inner; // i % inner without modulo
            let idx = *index.wrapping_add(i as usize);
            let src_pos = (outer * inner + idx) as usize;
            // Clamp to valid range: use idx directly (trust caller) but also handle simple flat case
            *output.wrapping_add(i as usize) = *input.wrapping_add(src_pos);
            i = i + 1;
        }
    }
}
