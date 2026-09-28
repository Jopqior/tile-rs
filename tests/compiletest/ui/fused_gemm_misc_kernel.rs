// build-pass

// Miscellaneous GEMM fusion kernels covering remaining fuse/ patterns.

#![feature(no_core)]

#![no_std]
#![no_core]

/// gemm + group_norm + min + bias_add
/// Maps to fuse/gemm_group_norm_min_bias_add.py
#[tile_std::tile_kernel]
pub fn fused_gemm_norm_min_bias(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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
        tile_std::__tile_mins_f32(buf_out, buf_out, 1.0f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_out, buf_out, 0.1f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}

/// matmul + avg + gelu + scale + max
/// Maps to fuse/matmul_avg_pool_gelu_scale_max.py (partial)
#[tile_std::tile_kernel]
pub fn fused_matmul_gelu_scale_max(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut buf2 = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // gelu: dst=buf2, src=buf (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf2, &buf, &mut tmp, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf2, buf2, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_maxs_f32(buf2, buf2, 0.0f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}

/// Min + softmax (for attention masking)
/// Maps to fuse/conv3d_min_softmax.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_min_softmax(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_mins_f32(buf, buf, 10.0f32, n);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=work, src=buf (destroyed), tmp
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// Group norm + tanh + hardswish + residual + log_sum_exp
/// Maps to fuse/conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp.py (partial)
#[tile_std::tile_kernel]
pub fn fused_norm_tanh_hardswish_residual_lse(
    x: *const f32, residual: *const f32, output: *mut f32, len: *const u32
) {
    unsafe {
        let n = *len;
        let mut bx = tile_std::__tile_buf_alloc(n);
        let br = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(br, residual, n);
        tile_std::__tile_pipe_barrier();

        // norm: dst=tmp, src=bx (preserved), work
        tile_std::kernel_ops::layernorm_f32(&mut tmp, &bx, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // tanh
        tile_std::kernel_ops::tanh_f32(tmp, tmp, n);
        tile_std::__tile_pipe_barrier();
        // hardswish: dst=bx, src=tmp (preserved), work
        tile_std::kernel_ops::hardswish_f32(&mut bx, &tmp, &mut work, n);
        tile_std::__tile_pipe_barrier();
        // residual add — br dead after
        tile_std::__tile_v_add_f32(br, bx, br, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, br, n);
    }
}

/// Group norm + scale + max_pool + clamp
/// Maps to fuse/conv2d_group_norm_scale_max_pool_clamp.py (partial)
#[tile_std::tile_kernel]
pub fn fused_norm_scale_clamp(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_out, buf_out, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardtanh_f32(buf_out, buf_out, -1.0f32, 1.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// Mean + norm (global average + normalization)
/// Maps to fuse/conv3d_group_norm_mean.py (partial)
#[tile_std::tile_kernel]
pub fn fused_norm_mean(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();

        // reduce_mean: dst=work, src=buf_out (preserved), tmp=buf (dead, needs mut)
        let mut buf2 = tile_std::__tile_buf_alloc(n);
        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf_out, &mut buf2, n);
        *output = mean;
    }
}

/// Activation + batch_norm + scaling (post-conv pattern)
/// Maps to fuse/conv2d_activation_batch_norm.py (partial)
#[tile_std::tile_kernel]
pub fn fused_relu_norm_scale(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::relu_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_out, buf_out, 2.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// Batch norm + scaling (pre-activation pattern)
/// Maps to fuse/conv2d_batch_norm_scaling.py (partial)
#[tile_std::tile_kernel]
pub fn fused_norm_scaling(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_out, buf_out, 3.14f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// Instance norm + divide
/// Maps to fuse/conv2d_instance_norm_divide.py (partial)
#[tile_std::tile_kernel]
pub fn fused_norm_divide(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_out, buf_out, 0.5f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// Group norm + min + clamp (with safety bounds)
/// Maps to fuse/conv3d_group_norm_min_clamp_dropout.py (partial)
#[tile_std::tile_kernel]
pub fn fused_norm_min_clamp(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_mins_f32(buf_out, buf_out, 5.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardtanh_f32(buf_out, buf_out, -3.0f32, 3.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
