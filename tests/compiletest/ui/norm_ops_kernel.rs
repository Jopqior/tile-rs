// build-pass

// Normalization operation kernels.
// Maps to MultiKernelBench/reference/normalization/ category.

#![feature(no_core)]

#![no_std]
#![no_core]

/// RMS Normalization: rms_norm(x) = x / sqrt(mean(x^2) + eps)
/// Maps to normalization/rms_norm.py
#[tile_std::tile_kernel]
pub fn rms_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let eps = 1e-5f32;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::rms_norm_f32(&mut buf_out, &buf_in, &mut buf_work, n, eps);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// L1 Norm: l1_norm(x) = sum(|x|)
/// Maps to normalization/l1_norm.py
/// Output is broadcast to a UB buffer and DMA-stored (scalar GM writes don't work on NPU).
#[tile_std::tile_kernel]
pub fn l1_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::kernel_ops::l1_norm_f32(&mut buf_work, &buf_in, &mut buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_work, buf_work, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_work, buf_work, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_work, n);
    }
}

/// L2 Norm (Frobenius for vectors): l2_norm(x) = sqrt(sum(x^2))
/// Maps to normalization/l2_norm.py and normalization/frobenius_norm.py
/// Output is broadcast to a UB buffer and DMA-stored (scalar GM writes don't work on NPU).
#[tile_std::tile_kernel]
pub fn l2_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::kernel_ops::l2_norm_f32(&mut buf_work, &buf_in, &mut buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_work, buf_work, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_work, buf_work, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_work, n);
    }
}

/// L2 Normalize: l2_normalize(x) = x / (l2_norm(x) + eps)
/// Maps to normalization/l2_norm.py (normalized variant)
#[tile_std::tile_kernel]
pub fn l2_normalize(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let eps = 1e-8f32;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::l2_normalize_f32(&mut buf_out, &buf_in, &mut buf_work, n, eps);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// Layer Normalization (already in composite_ops_kernel.rs, adding for completeness)
/// Maps to normalization/layer_norm.py
#[tile_std::tile_kernel]
pub fn layer_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let eps = 1e-5f32;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf_in, &mut buf_work, n, eps);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
