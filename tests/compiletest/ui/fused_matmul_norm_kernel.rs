// build-pass

// Fused matmul + normalization + activation kernels.
// Maps to MultiKernelBench/reference/fuse/ category (gemm_*_norm_* entries).

#![feature(no_core)]

#![no_std]
#![no_core]

/// gemm + batch_norm + gelu (simplified: matmul + layernorm + gelu)
/// Maps to fuse/gemm_batch_norm_gelu_group_norm_mean_relu.py
#[tile_std::tile_kernel]
pub fn fused_gemm_norm_gelu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=work, src=buf_out (preserved), tmp=buf (dead)
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::kernel_ops::gelu_f32(&mut work, &buf_out, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

/// gemm + batch_norm + scaling + softmax
/// Maps to fuse/gemm_batch_norm_scaling_softmax.py
#[tile_std::tile_kernel]
pub fn fused_gemm_norm_scale_softmax(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // norm
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // scale
        tile_std::__tile_muls_f32(buf_out, buf_out, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=work, src=buf_out (destroyed), tmp
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf_out, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

/// gemm + scale + batch_norm
/// Maps to fuse/gemm_scale_batch_norm.py
#[tile_std::tile_kernel]
pub fn fused_gemm_scale_norm(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // scale
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        // norm
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}

/// gemm + group_norm + hardtanh
/// Maps to fuse/gemm_group_norm_hardtanh.py
#[tile_std::tile_kernel]
pub fn fused_gemm_norm_hardtanh(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardtanh_f32(buf_out, buf_out, -1.0f32, 1.0f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}

/// gemm + group_norm + swish + multiply + swish
/// Maps to fuse/gemm_group_norm_swish_multiply_swish.py
#[tile_std::tile_kernel]
pub fn fused_gemm_norm_swish_mul_swish(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // norm
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // swish: dst=work, src=buf_out (preserved), tmp
        tile_std::kernel_ops::swish_f32(&mut work, &buf_out, &mut tmp, total);
        tile_std::__tile_pipe_barrier();
        // multiply by 2
        tile_std::__tile_muls_f32(work, work, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        // swish again: dst=buf_out, src=work (preserved), tmp
        tile_std::kernel_ops::swish_f32(&mut buf_out, &work, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}

/// gemm + bias + hardtanh + mish + group_norm
/// Maps to fuse/gemm_bias_add_hardtanh_mish_group_norm.py
#[tile_std::tile_kernel]
pub fn fused_gemm_bias_hardtanh_mish_norm(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // bias add
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        // hardtanh
        tile_std::kernel_ops::hardtanh_f32(buf, buf, -1.0f32, 1.0f32, total);
        tile_std::__tile_pipe_barrier();
        // mish: dst=buf_out, src=buf (preserved), work
        tile_std::kernel_ops::mish_f32(&mut buf_out, &buf, &mut work, total);
        tile_std::__tile_pipe_barrier();
        // norm: dst=work, src=buf_out (preserved), tmp=buf (dead)
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf_out, &mut tmp, total, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

// === Split variants for 1:1 MKB kernel mapping ===

/// gemm + scale + batch_norm (same as fused_gemm_scale_norm)
/// Maps to fuse/gemm_scale_batch_norm.py
#[tile_std::tile_kernel]
pub fn gemm_scale_batch_norm(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(buf, buf, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}

/// gemm + scale + batchnorm (variant naming)
/// Maps to fuse/gemm_scale_batchnorm.py
#[tile_std::tile_kernel]
pub fn gemm_scale_batchnorm(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(buf, buf, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}
