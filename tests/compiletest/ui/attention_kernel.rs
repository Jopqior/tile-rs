// build-pass

// Attention-related kernels.
// Maps to MultiKernelBench/reference/attention/ category.
// Implements the core element-wise operations used in attention mechanisms.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Scaled dot-product attention scores: scores = softmax(Q*K^T / sqrt(d))
/// Simplified to: softmax(x / sqrt(d)) on a pre-computed QK^T vector.
/// Maps to attention/ category (attention score normalization part)
#[tile_std::tile_kernel]
pub fn attention_softmax(input: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let d_model = *config;
        let scale = 1.0f32 / tile_std::core::builtins::sqrtf(d_model);

        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // scale
        tile_std::__tile_muls_f32(buf, buf, scale, n);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=work, src=buf (destroyed), work=... need extra buf
        let mut tmp = tile_std::__tile_buf_alloc(n);
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// Residual add + layer norm (common transformer pattern):
///   output = layernorm(x + residual)
#[tile_std::tile_kernel]
pub fn residual_add_layernorm(x: *const f32, residual: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let eps = 1e-5f32;
        let mut bx = tile_std::__tile_buf_alloc(n);
        let br = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(br, residual, n);
        tile_std::__tile_pipe_barrier();

        // x + residual → br dead after, reuse as output
        tile_std::__tile_v_add_f32(br, bx, br, n);
        tile_std::__tile_pipe_barrier();
        // layernorm: src=br, dst=bx (distinct buffers)
        tile_std::kernel_ops::layernorm_f32(&mut bx, &br, &mut work, n, eps);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bx, n);
    }
}

/// Residual add + rms norm:
///   output = rms_norm(x + residual)
#[tile_std::tile_kernel]
pub fn residual_add_rmsnorm(x: *const f32, residual: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let eps = 1e-5f32;
        let mut bx = tile_std::__tile_buf_alloc(n);
        let br = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(br, residual, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_add_f32(br, bx, br, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::rms_norm_f32(&mut bx, &br, &mut work, n, eps);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bx, n);
    }
}

/// SwiGLU activation (used in LLaMA/Mistral):
///   swiglu(x, gate) = swish(gate) * x
#[tile_std::tile_kernel]
pub fn swiglu(x: *const f32, gate: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bg, gate, n);
        tile_std::__tile_pipe_barrier();

        // swish(gate) = gate * sigmoid(gate) — src preserved, result in tmp
        tile_std::kernel_ops::swish_f32(&mut tmp, &bg, &mut work, n);
        tile_std::__tile_pipe_barrier();
        // swiglu = swish(gate) * x
        tile_std::__tile_v_mul_f32(work, bx, tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// GeGLU activation: geglu(x, gate) = gelu(gate) * x
#[tile_std::tile_kernel]
pub fn geglu(x: *const f32, gate: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bg, gate, n);
        tile_std::__tile_pipe_barrier();

        // gelu: src preserved, result in tmp
        tile_std::kernel_ops::gelu_f32(&mut tmp, &bg, &mut work, n);
        tile_std::__tile_pipe_barrier();
        // geglu = gelu(gate) * x
        tile_std::__tile_v_mul_f32(work, bx, tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// Masked fill: output = where(mask > 0, x, fill_value)
/// Approximate: output[i] = x[i] * mask[i] + fill * (1 - mask[i])
/// where mask is 0 or 1
#[tile_std::tile_kernel]
pub fn masked_fill(x: *const f32, mask: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let fill_value = *config;
        let bx = tile_std::__tile_buf_alloc(n);
        let bm = tile_std::__tile_buf_alloc(n);
        let bt = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bm, mask, n);
        tile_std::__tile_pipe_barrier();

        // bt = x * mask (keep values where mask=1)
        tile_std::__tile_v_mul_f32(bt, bx, bm, n);
        tile_std::__tile_pipe_barrier();

        // bm = 1 - mask
        tile_std::__tile_muls_f32(bm, bm, -1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bm, bm, 1.0f32, n);
        tile_std::__tile_pipe_barrier();

        // bm = fill_value * (1 - mask)
        tile_std::__tile_muls_f32(bm, bm, fill_value, n);
        tile_std::__tile_pipe_barrier();

        // output = x*mask + fill*(1-mask)
        tile_std::__tile_v_add_f32(bt, bt, bm, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bt, n);
    }
}
