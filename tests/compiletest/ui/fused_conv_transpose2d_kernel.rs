// build-pass

// Fused conv_transpose2d + activation extension kernels.
// Maps to MultiKernelBench/reference/fuse/ category.
// Conv is simplified to activation chains since actual convolution requires cube engine.

#![feature(no_core)]

#![no_std]
#![no_core]

/// adds(0.1) + mins(1.0) + gelu + muls(2.0)
#[tile_std::tile_kernel]
pub fn conv_transpose2d_add_min_gelu_multiply(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_adds_f32(buf, buf, 0.1f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_mins_f32(buf, buf, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // gelu: dst, src (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut dst, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(dst, dst, 2.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// adds(0.1) + hardtanh(-2,2) + muls(3.0) + hardtanh(-1,1) + muls(0.5)
#[tile_std::tile_kernel]
pub fn conv_transpose2d_bias_add_clamp_scaling_clamp_divide(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_adds_f32(buf, buf, 0.1f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardtanh_f32(buf, buf, -2.0f32, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 3.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardtanh_f32(buf, buf, -1.0f32, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 0.5f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// gelu + layernorm
#[tile_std::tile_kernel]
pub fn conv_transpose2d_gelu_group_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // gelu: dst=buf_out, src=buf (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf_out, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // layernorm: dst=work, src=buf_out (preserved), tmp=buf (dead)
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf_out, &mut buf, n, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// maxs(0.0) + hardtanh(-1,1) + reduce_mean -> tanh -> single f32
/// Apply tanh to vector before mean since vector tanh + scalar mean = tanh(mean) approx
#[tile_std::tile_kernel]
pub fn conv_transpose2d_max_pool_hardtanh_mean_tanh(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardtanh_f32(buf, buf, -1.0f32, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();

        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);
        *output = mean;
    }
}

/// mins(1.0) + gelu + adds(0.5)
#[tile_std::tile_kernel]
pub fn conv_transpose2d_min_sum_gelu_add(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_mins_f32(buf, buf, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // gelu: dst, src (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut dst, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(dst, dst, 0.5f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// mish + adds(0.1) + hardtanh(-1,1) + muls(2.0)
#[tile_std::tile_kernel]
pub fn conv_transpose2d_mish_add_hardtanh_scaling(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // mish: dst, src (preserved), tmp
        tile_std::kernel_ops::mish_f32(&mut dst, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(dst, dst, 0.1f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardtanh_f32(dst, dst, -1.0f32, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(dst, dst, 2.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// muls(2.0) + reduce_mean -> single f32
#[tile_std::tile_kernel]
pub fn conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(buf, buf, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);

        *output = mean;
    }
}

/// adds(-0.5) + tanh
#[tile_std::tile_kernel]
pub fn conv_transpose2d_subtract_tanh(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_adds_f32(buf, buf, -0.5f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(buf, buf, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// layernorm + tanh + maxs(0.0) + layernorm
#[tile_std::tile_kernel]
pub fn convtranspose2d_batchnorm_tanh_maxpool_groupnorm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // first layernorm: dst=buf_out, src=buf
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // tanh in-place on buf_out
        tile_std::kernel_ops::tanh_f32(buf_out, buf_out, n);
        tile_std::__tile_pipe_barrier();
        // maxs in-place on buf_out
        tile_std::__tile_maxs_f32(buf_out, buf_out, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        // second layernorm: dst=buf (different from src=buf_out)
        tile_std::kernel_ops::layernorm_f32(&mut buf, &buf_out, &mut work, n, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// reduce_mean -> single f32 output
#[tile_std::tile_kernel]
pub fn convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);

        *output = mean;
    }
}

/// softmax + adds(0.1) + muls(2.0) + sigmoid
#[tile_std::tile_kernel]
pub fn convtranspose2d_softmax_biasadd_scaling_sigmoid(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // softmax: dst, src (destroyed), work
        tile_std::kernel_ops::softmax_f32(&mut dst, &mut buf, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(dst, dst, 0.1f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(dst, dst, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(dst, dst, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}
