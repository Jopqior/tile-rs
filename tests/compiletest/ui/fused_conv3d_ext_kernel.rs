// build-pass

// Fused conv3d + activation extension kernels.
// Maps to MultiKernelBench/reference/fuse/ category (conv3d_* entries).
// Conv3d is simplified to norm/activation chains since actual convolution requires cube engine.

#![feature(no_core)]

#![no_std]
#![no_core]

/// divide + max + global_avg_pool + bias_add + sum
/// Maps to fuse/conv3d_divide_max_global_avg_pool_bias_add_sum.py
/// muls(0.5) + maxs(0.0) + reduce_mean → single f32
#[tile_std::tile_kernel]
pub fn conv3d_divide_max_global_avg_pool_bias_add_sum(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // divide by 2
        tile_std::__tile_muls_f32(buf, buf, 0.5f32, n);
        tile_std::__tile_pipe_barrier();
        // max with 0
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        // reduce mean → single f32
        let result = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);

        *output = result;
    }
}

/// leaky_relu + sum + clamp + gelu
/// Maps to fuse/conv3d_leaky_relu_sum_clamp_gelu.py
/// leaky_relu(0.01) + hardtanh(-2,2) + gelu
#[tile_std::tile_kernel]
pub fn conv3d_leaky_relu_sum_clamp_gelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // leaky relu: dst=work, src=buf (destroyed), tmp
        tile_std::kernel_ops::leaky_relu_f32(&mut work, &mut buf, &mut tmp, 0.01f32, n);
        tile_std::__tile_pipe_barrier();
        // clamp to [-2, 2]
        tile_std::kernel_ops::hardtanh_f32(work, work, -2.0f32, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=buf, src=work (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf, &work, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// multiply + instance_norm + clamp + multiply + max
/// Maps to fuse/conv3d_multiply_instance_norm_clamp_multiply_max.py
/// muls(2.0) + layernorm + hardtanh(-1,1) + muls(3.0) + maxs(0.0)
#[tile_std::tile_kernel]
pub fn conv3d_multiply_instance_norm_clamp_multiply_max(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // multiply by 2
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // layernorm: dst != src
        tile_std::kernel_ops::layernorm_f32(&mut dst, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // clamp to [-1, 1]
        tile_std::kernel_ops::hardtanh_f32(dst, dst, -1.0f32, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // multiply by 3
        tile_std::__tile_muls_f32(dst, dst, 3.0f32, n);
        tile_std::__tile_pipe_barrier();
        // max with 0
        tile_std::__tile_maxs_f32(dst, dst, 0.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// relu + leaky_relu + gelu + sigmoid + bias_add
/// Maps to fuse/conv3d_relu_leaky_relu_gelu_sigmoid_bias_add.py
/// relu + leaky_relu(0.01) + gelu + sigmoid + adds(0.1)
#[tile_std::tile_kernel]
pub fn conv3d_relu_leaky_relu_gelu_sigmoid_bias_add(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // relu
        tile_std::kernel_ops::relu_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // leaky relu: dst=work, src=buf (destroyed), tmp
        tile_std::kernel_ops::leaky_relu_f32(&mut work, &mut buf, &mut tmp, 0.01f32, n);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=buf, src=work (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf, &work, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // sigmoid
        tile_std::kernel_ops::sigmoid_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // bias add (scalar)
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// scaling + tanh + multiply + sigmoid
/// Maps to fuse/conv3d_scaling_tanh_multiply_sigmoid.py
/// muls(2.0) + tanh + sigmoid
#[tile_std::tile_kernel]
pub fn conv3d_scaling_tanh_multiply_sigmoid(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // scale by 2
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // tanh
        tile_std::kernel_ops::tanh_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // sigmoid
        tile_std::kernel_ops::sigmoid_f32(buf, buf, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// softmax + max_pool + max_pool
/// Maps to fuse/conv3d_softmax_max_pool_max_pool.py
/// softmax + maxs(0.0) + maxs(-0.5)
#[tile_std::tile_kernel]
pub fn conv3d_softmax_max_pool_max_pool(input: *const f32, output: *mut f32, len: *const u32) {
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
        // max pool (simplified as maxs with threshold)
        tile_std::__tile_maxs_f32(dst, dst, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        // max pool again
        tile_std::__tile_maxs_f32(dst, dst, -0.5f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}
