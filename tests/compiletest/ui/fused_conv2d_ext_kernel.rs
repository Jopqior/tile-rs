// build-pass

// Fused conv2d + activation extension kernels.
// Maps to MultiKernelBench/reference/fuse/ category (conv2d_* entries).
// Conv2d is simplified to norm (layernorm) since actual convolution requires cube engine.

#![feature(no_core)]

#![no_std]
#![no_core]

/// conv2d + activation + batch_norm
/// Unary: relu + layernorm + scale(2.0)
/// Maps to fuse/conv2d_activation_batch_norm.py
#[tile_std::tile_kernel]
pub fn conv2d_activation_batch_norm(input: *const f32, output: *mut f32, len: *const u32) {
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

/// conv2d + add + scale + sigmoid + group_norm
/// Unary: adds(0.1) + muls(2.0) + sigmoid + layernorm
/// Maps to fuse/conv2d_add_scale_sigmoid_group_norm.py
#[tile_std::tile_kernel]
pub fn conv2d_add_scale_sigmoid_group_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_adds_f32(buf, buf, 0.1f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// conv2d + avg_pool + sigmoid + sum
/// Unary: sigmoid + reduce_sum (write single f32)
/// Maps to fuse/conv2d_avg_pool_sigmoid_sum.py
#[tile_std::tile_kernel]
pub fn conv2d_avg_pool_sigmoid_sum(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::sigmoid_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();

        let sum = tile_std::__tile_v_reduce_sum_f32(buf, buf, work, n);
        *output = sum;
    }
}

/// conv2d + batch_norm + scaling
/// Unary: layernorm + muls(3.14)
/// Maps to fuse/conv2d_batch_norm_scaling.py
#[tile_std::tile_kernel]
pub fn conv2d_batch_norm_scaling(input: *const f32, output: *mut f32, len: *const u32) {
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

/// conv2d + gelu + global_avg_pool
/// Unary: gelu + reduce_mean (write single f32)
/// Maps to fuse/conv2d_gelu_global_avg_pool.py
#[tile_std::tile_kernel]
pub fn conv2d_gelu_global_avg_pool(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // gelu: dst=buf_out, src=buf (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf_out, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();

        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf_out, &mut tmp, n);
        *output = mean;
    }
}

/// conv2d + group_norm + scale + max_pool + clamp
/// Unary: layernorm + muls(2.0) + hardtanh(-1,1)
/// Maps to fuse/conv2d_group_norm_scale_max_pool_clamp.py
#[tile_std::tile_kernel]
pub fn conv2d_group_norm_scale_max_pool_clamp(input: *const f32, output: *mut f32, len: *const u32) {
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

/// conv2d + group_norm + tanh + hard_swish + residual_add + log_sum_exp
/// Binary (x, residual): layernorm + tanh + hardswish + add residual
/// Maps to fuse/conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp.py
#[tile_std::tile_kernel]
pub fn conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp(
    x: *const f32, residual: *const f32, output: *mut f32, len: *const u32
) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let mut bx_out = tile_std::__tile_buf_alloc(n);
        let br = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(br, residual, n);
        tile_std::__tile_pipe_barrier();

        // layernorm (dst != src)
        tile_std::kernel_ops::layernorm_f32(&mut bx_out, &bx, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // tanh
        tile_std::kernel_ops::tanh_f32(bx_out, bx_out, n);
        tile_std::__tile_pipe_barrier();
        // hardswish: dst=work, src=bx_out (preserved), tmp
        tile_std::kernel_ops::hardswish_f32(&mut work, &bx_out, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // residual add — use bx (dead after layernorm) as distinct dst
        tile_std::__tile_v_add_f32(bx, work, br, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bx, n);
    }
}

/// conv2d + instance_norm + divide
/// Unary: layernorm + muls(0.5)
/// Maps to fuse/conv2d_instance_norm_divide.py
#[tile_std::tile_kernel]
pub fn conv2d_instance_norm_divide(input: *const f32, output: *mut f32, len: *const u32) {
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

/// conv2d + subtract + hard_swish + max_pool + mish
/// Unary: adds(-0.5) + hardswish + mish
/// Maps to fuse/conv2d_subtract_hard_swish_max_pool_mish.py
#[tile_std::tile_kernel]
pub fn conv2d_subtract_hard_swish_max_pool_mish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut tmp2 = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_adds_f32(buf, buf, -0.5f32, n);
        tile_std::__tile_pipe_barrier();
        // hardswish: dst, src (preserved), tmp
        tile_std::kernel_ops::hardswish_f32(&mut dst, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // mish: dst=tmp2, src=dst (preserved), tmp=buf (dead)
        tile_std::kernel_ops::mish_f32(&mut tmp2, &dst, &mut buf, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, tmp2, n);
    }
}

/// conv2d + subtract + subtract + mish
/// Unary: adds(-0.3) + adds(-0.2) + mish
/// Maps to fuse/conv2d_subtract_subtract_mish.py
#[tile_std::tile_kernel]
pub fn conv2d_subtract_subtract_mish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_adds_f32(buf, buf, -0.3f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, -0.2f32, n);
        tile_std::__tile_pipe_barrier();
        // mish: dst, src (preserved), tmp
        tile_std::kernel_ops::mish_f32(&mut dst, &buf, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// conv2d + subtract + tanh + subtract + avg_pool
/// Unary: adds(-0.5) + tanh + adds(-0.1) + reduce_mean (single f32)
/// Maps to fuse/conv2d_subtract_tanh_subtract_avg_pool.py
#[tile_std::tile_kernel]
pub fn conv2d_subtract_tanh_subtract_avg_pool(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_adds_f32(buf, buf, -0.5f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, -0.1f32, n);
        tile_std::__tile_pipe_barrier();

        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);
        *output = mean;
    }
}
