// build-pass

// Extended normalization operations.
// Maps to MultiKernelBench/reference/normalization/ category.
// Covers batch_norm, group_norm, instance_norm, frobenius_norm.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
/// Simplified to element-wise form (per-channel stats pre-computed).
#[tile_std::tile_kernel]
pub fn batch_norm(input: *const f32, mean: *const f32, var: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bm = tile_std::__tile_buf_alloc(n);
        let bv = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, input, n);
        tile_std::__tile_buf_load_f32(bm, mean, n);
        tile_std::__tile_buf_load_f32(bv, var, n);
        tile_std::__tile_pipe_barrier();
        // x - mean → bm dead after
        tile_std::__tile_v_sub_f32(bm, bx, bm, n);
        tile_std::__tile_pipe_barrier();
        // var + eps
        tile_std::__tile_adds_f32(bv, bv, 1e-5f32, n);
        tile_std::__tile_pipe_barrier();
        // 1/sqrt(var+eps)
        tile_std::__tile_v_sqrt_f32(bv, bv, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_reciprocal_f32(bv, bv, n);
        tile_std::__tile_pipe_barrier();
        // (x - mean) / sqrt(var + eps) → bv dead after
        tile_std::__tile_v_mul_f32(bx, bm, bv, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bx, n);
    }
}

/// Group normalization: normalize within groups (simplified as full norm)
#[tile_std::tile_kernel]
pub fn group_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, out, n);
    }
}

/// Instance normalization: normalize per-instance (same as layernorm for 1D)
#[tile_std::tile_kernel]
pub fn instance_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, out, n);
    }
}

/// Frobenius norm: sqrt(sum(x^2))
#[tile_std::tile_kernel]
pub fn frobenius_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        // x^2
        tile_std::__tile_v_mul_f32(buf, buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // sum(x^2)
        let sum_sq = tile_std::__tile_v_reduce_sum_f32(buf, buf, tmp, n);
        // sqrt(sum(x^2))
        *output = tile_std::core::builtins::sqrtf(sum_sq);
    }
}
