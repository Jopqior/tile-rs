// build-pass

// Architecture-level operation kernels.
// Maps to MultiKernelBench/reference/arch/ category.
// These are building blocks used in neural network architectures
// (MLP layers, attention blocks, feed-forward networks).

#![feature(no_core)]

#![no_std]
#![no_core]

/// MLP block: relu(matmul(x, W))
/// Common pattern in feed-forward networks
#[tile_std::tile_kernel]
pub fn mlp_relu(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// MLP block: gelu(matmul(x, W) + b)
/// GPT-style MLP with bias
#[tile_std::tile_kernel]
pub fn mlp_gelu_bias(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::gelu_f32(&mut tmp, &buf, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, tmp, total);
    }
}

/// MLP block: swish(matmul(x, W))
/// LLaMA-style MLP
#[tile_std::tile_kernel]
pub fn mlp_swish(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::swish_f32(&mut tmp, &buf, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, tmp, total);
    }
}

/// FFN block: matmul + norm + activation
/// Transformer feed-forward with pre-norm
#[tile_std::tile_kernel]
pub fn ffn_prenorm(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::gelu_f32(&mut extra, &buf_out, &mut work, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, extra, total);
    }
}

/// Down-projection: scale(matmul(x, W))
#[tile_std::tile_kernel]
pub fn down_proj(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// Attention score normalization: softmax(x / sqrt(d_k))
#[tile_std::tile_kernel]
pub fn attention_score_norm(input: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let d_k = *config;
        let scale = 1.0f32 / tile_std::core::builtins::sqrtf(d_k);
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut extra, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// RoPE frequency computation: freq = 1 / (base^(2i/d))
/// Simplified: compute exponential decay of frequencies
#[tile_std::tile_kernel]
pub fn rope_freq(output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let base = *config;
        let buf = tile_std::__tile_buf_alloc(n);

        // Generate indices: 0, 2, 4, ... (even dims)
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let dim_frac = (2 * i) as f32 / (n as f32);
            // freq_i = 1 / base^dim_frac ≈ exp(-dim_frac * ln(base))
            let log_base = tile_std::core::builtins::logf(base);
            let freq = tile_std::core::builtins::expf(-dim_frac * log_base);
            *output.wrapping_add(i as usize) = freq;
            i = i + 1;
        }
    }
}

/// Embedding lookup (simplified: scale input)
#[tile_std::tile_kernel]
pub fn embedding_scale(input: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// Layer output: sigmoid_gate * value + residual
#[tile_std::tile_kernel]
pub fn gated_residual(value: *const f32, gate: *const f32, residual: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bv = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        let br = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bv, value, n);
        tile_std::__tile_buf_load_f32(bg, gate, n);
        tile_std::__tile_buf_load_f32(br, residual, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::sigmoid_f32(bg, bg, n);
        tile_std::__tile_pipe_barrier();
        // bg dead after mul, br dead after add
        tile_std::__tile_v_mul_f32(bg, bv, bg, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(br, bg, br, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, br, n);
    }
}

/// Scaled dot product (no softmax): q * k * scale
#[tile_std::tile_kernel]
pub fn scaled_dot(q: *const f32, k: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bq = tile_std::__tile_buf_alloc(n);
        let bk = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bk, k, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_mul_f32(bk, bq, bk, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bk, bk, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bk, n);
    }
}

/// Final projection: matmul + bias + sigmoid (classifier head)
#[tile_std::tile_kernel]
pub fn classifier_head(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// Regression head: matmul + bias (no activation)
#[tile_std::tile_kernel]
pub fn regression_head(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, 0.01f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// Softmax classifier: matmul + softmax
#[tile_std::tile_kernel]
pub fn softmax_classifier(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, work, total);
    }
}

// === Split variants for 1:1 MKB kernel mapping ===

/// MLP block: relu(matmul(x, W))
#[tile_std::tile_kernel]
pub fn mlp(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// Deep narrow MLP block: relu(matmul(x, W))
#[tile_std::tile_kernel]
pub fn deep_narrow_mlp(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// Shallow wide MLP block: relu(matmul(x, W))
#[tile_std::tile_kernel]
pub fn shallow_wide_mlp(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}
