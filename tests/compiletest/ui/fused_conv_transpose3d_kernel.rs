// build-pass

// Fused conv_transpose3d + activation extension kernels.
// Maps to MultiKernelBench/reference/fuse/ category (conv_transpose3d_* entries).
// Conv is simplified to activation chains since actual convolution requires cube engine.

#![feature(no_core)]

#![no_std]
#![no_core]

/// add + hard_swish
/// Maps to fuse/conv_transpose3d_add_hard_swish.py
/// adds(0.1) + hardswish
#[tile_std::tile_kernel]
pub fn conv_transpose3d_add_hard_swish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // bias add 0.1
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, n);
        tile_std::__tile_pipe_barrier();
        // hardswish: dst, src (preserved), tmp must all be distinct
        tile_std::kernel_ops::hardswish_f32(&mut dst, &buf, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// avg_pool + clamp + softmax + multiply
/// Maps to fuse/conv_transpose3d_avg_pool_clamp_softmax_multiply.py
/// hardtanh(-2,2) + softmax + muls(2.0)
#[tile_std::tile_kernel]
pub fn conv_transpose3d_avg_pool_clamp_softmax_multiply(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // clamp to [-2, 2]
        tile_std::kernel_ops::hardtanh_f32(buf, buf, -2.0f32, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // softmax: dst, src (destroyed), work must all be distinct
        tile_std::kernel_ops::softmax_f32(&mut dst, &mut buf, &mut work, n);
        tile_std::__tile_pipe_barrier();
        // multiply by 2
        tile_std::__tile_muls_f32(dst, dst, 2.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// batch_norm + avg_pool + avg_pool
/// Maps to fuse/conv_transpose3d_batch_norm_avg_pool_avg_pool.py
/// layernorm + reduce_mean → single f32
#[tile_std::tile_kernel]
pub fn conv_transpose3d_batch_norm_avg_pool_avg_pool(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // layernorm: dst != src
        tile_std::kernel_ops::layernorm_f32(&mut dst, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // reduce mean → single f32
        let result = tile_std::kernel_ops::reduce_mean_f32(&mut work, &dst, &mut tmp, n);

        *output = result;
    }
}

/// batch_norm + subtract
/// Maps to fuse/conv_transpose3d_batch_norm_subtract.py
/// layernorm + adds(-0.5)
#[tile_std::tile_kernel]
pub fn conv_transpose3d_batch_norm_subtract(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // layernorm: dst != src
        tile_std::kernel_ops::layernorm_f32(&mut dst, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // subtract 0.5
        tile_std::__tile_adds_f32(dst, dst, -0.5f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// clamp_min + divide
/// Maps to fuse/conv_transpose3d_clamp_min_divide.py
/// hardtanh(-1,1) + mins(0.5) + muls(0.5)
#[tile_std::tile_kernel]
pub fn conv_transpose3d_clamp_min_divide(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // clamp to [-1, 1]
        tile_std::kernel_ops::hardtanh_f32(buf, buf, -1.0f32, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // min with 0.5
        tile_std::__tile_mins_f32(buf, buf, 0.5f32, n);
        tile_std::__tile_pipe_barrier();
        // divide by 2
        tile_std::__tile_muls_f32(buf, buf, 0.5f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// layer_norm + gelu + scaling
/// Maps to fuse/conv_transpose3d_layer_norm_gelu_scaling.py
/// layernorm + gelu + muls(2.0)
#[tile_std::tile_kernel]
pub fn conv_transpose3d_layer_norm_gelu_scaling(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // layernorm: dst != src
        tile_std::kernel_ops::layernorm_f32(&mut dst, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=work, src=dst (preserved), tmp=buf (dead)
        tile_std::kernel_ops::gelu_f32(&mut work, &dst, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // scale by 2
        tile_std::__tile_muls_f32(work, work, 2.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// leaky_relu + multiply + leaky_relu + max
/// Maps to fuse/conv_transpose3d_leaky_relu_multiply_leaky_relu_max.py
/// leaky_relu(0.01) + muls(2.0) + leaky_relu(0.01) + maxs(0.0)
#[tile_std::tile_kernel]
pub fn conv_transpose3d_leaky_relu_multiply_leaky_relu_max(input: *const f32, output: *mut f32, len: *const u32) {
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
        // multiply by 2
        tile_std::__tile_muls_f32(work, work, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // leaky relu again: dst=buf, src=work (destroyed), tmp
        tile_std::kernel_ops::leaky_relu_f32(&mut buf, &mut work, &mut tmp, 0.01f32, n);
        tile_std::__tile_pipe_barrier();
        // max with 0
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// log_sum_exp + hard_swish + subtract + clamp_max
/// Maps to fuse/conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max.py
/// hardswish + adds(-0.5) + hardtanh(-1,1) + maxs(0.0)
#[tile_std::tile_kernel]
pub fn conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // hardswish: dst, src (preserved), tmp
        tile_std::kernel_ops::hardswish_f32(&mut dst, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // subtract 0.5
        tile_std::__tile_adds_f32(dst, dst, -0.5f32, n);
        tile_std::__tile_pipe_barrier();
        // clamp to [-1, 1]
        tile_std::kernel_ops::hardtanh_f32(dst, dst, -1.0f32, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // max with 0
        tile_std::__tile_maxs_f32(dst, dst, 0.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// max + max + sum
/// Maps to fuse/conv_transpose3d_max_max_sum.py
/// maxs(0.0) + maxs(-0.5) + reduce_sum → single f32
#[tile_std::tile_kernel]
pub fn conv_transpose3d_max_max_sum(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // max with 0
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        // max with -0.5
        tile_std::__tile_maxs_f32(buf, buf, -0.5f32, n);
        tile_std::__tile_pipe_barrier();
        // reduce sum → single f32
        let result = tile_std::__tile_v_reduce_sum_f32(work, buf, tmp, n);

        *output = result;
    }
}

/// max_pool + softmax + subtract + swish + max
/// Maps to fuse/conv_transpose3d_max_pool_softmax_subtract_swish_max.py
/// maxs(0.0) + softmax + adds(-0.1) + swish + maxs(0.0)
#[tile_std::tile_kernel]
pub fn conv_transpose3d_max_pool_softmax_subtract_swish_max(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // max with 0
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=work, src=buf (destroyed), tmp
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // subtract 0.1
        tile_std::__tile_adds_f32(work, work, -0.1f32, n);
        tile_std::__tile_pipe_barrier();
        // swish: dst=buf, src=work (preserved), tmp
        tile_std::kernel_ops::swish_f32(&mut buf, &work, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // max with 0
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// multiply + max + global_avg_pool + clamp
/// Maps to fuse/conv_transpose3d_multiply_max_global_avg_pool_clamp.py
/// muls(2.0) + maxs(0.0) + hardtanh(-1,1)
#[tile_std::tile_kernel]
pub fn conv_transpose3d_multiply_max_global_avg_pool_clamp(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // multiply by 2
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // max with 0
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        // clamp to [-1, 1]
        tile_std::kernel_ops::hardtanh_f32(buf, buf, -1.0f32, 1.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// scale + batch_norm + global_avg_pool
/// Maps to fuse/conv_transpose3d_scale_batch_norm_global_avg_pool.py
/// muls(2.0) + layernorm + reduce_mean → single f32
#[tile_std::tile_kernel]
pub fn conv_transpose3d_scale_batch_norm_global_avg_pool(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // scale by 2
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // layernorm: dst != src
        tile_std::kernel_ops::layernorm_f32(&mut dst, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // reduce mean → single f32
        let result = tile_std::kernel_ops::reduce_mean_f32(&mut work, &dst, &mut tmp, n);

        *output = result;
    }
}

/// scaling + avg_pool + bias_add + scaling
/// Maps to fuse/conv_transpose3d_scaling_avg_pool_bias_add_scaling.py
/// muls(2.0) + adds(0.1) + muls(3.0)
#[tile_std::tile_kernel]
pub fn conv_transpose3d_scaling_avg_pool_bias_add_scaling(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // scale by 2
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // bias add 0.1
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, n);
        tile_std::__tile_pipe_barrier();
        // scale by 3
        tile_std::__tile_muls_f32(buf, buf, 3.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// softmax + sigmoid
/// Maps to fuse/conv_transpose3d_softmax_sigmoid.py
/// softmax + sigmoid
#[tile_std::tile_kernel]
pub fn conv_transpose3d_softmax_sigmoid(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // softmax: dst, src (destroyed), work must all be distinct
        tile_std::kernel_ops::softmax_f32(&mut dst, &mut buf, &mut work, n);
        tile_std::__tile_pipe_barrier();
        // sigmoid
        tile_std::kernel_ops::sigmoid_f32(dst, dst, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}

/// sum + layer_norm + avg_pool + gelu
/// Maps to fuse/conv_transpose3d_sum_layer_norm_avg_pool_gelu.py
/// layernorm + gelu
#[tile_std::tile_kernel]
pub fn conv_transpose3d_sum_layer_norm_avg_pool_gelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // layernorm: dst != src
        tile_std::kernel_ops::layernorm_f32(&mut dst, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=work, src=dst (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut work, &dst, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// sum + residual_add + multiply + residual_add (Binary)
/// Maps to fuse/conv_transpose3d_sum_residual_add_multiply_residual_add.py
/// add(x, residual) + muls(2.0) + add(residual) again
#[tile_std::tile_kernel]
pub fn conv_transpose3d_sum_residual_add_multiply_residual_add(x: *const f32, residual: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let br = tile_std::__tile_buf_alloc(n);
        let btmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(br, residual, n);
        tile_std::__tile_pipe_barrier();

        // x + residual → btmp (3 distinct buffers)
        tile_std::__tile_v_add_f32(btmp, bx, br, n);
        tile_std::__tile_pipe_barrier();
        // multiply by 2 (scalar op, in-place OK)
        tile_std::__tile_muls_f32(btmp, btmp, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // add residual again: bx is free, use as output (3 distinct: bx, btmp, br)
        tile_std::__tile_v_add_f32(bx, btmp, br, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bx, n);
    }
}

/// swish + group_norm + hard_swish
/// Maps to fuse/conv_transpose3d_swish_group_norm_hard_swish.py
/// swish + layernorm + hardswish
#[tile_std::tile_kernel]
pub fn conv_transpose3d_swish_group_norm_hard_swish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // swish: dst=tmp, src=buf (preserved), work
        tile_std::kernel_ops::swish_f32(&mut tmp, &buf, &mut work, n);
        tile_std::__tile_pipe_barrier();
        // layernorm: dst=dst, src=tmp (preserved), work=buf (dead)
        tile_std::kernel_ops::layernorm_f32(&mut dst, &tmp, &mut buf, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // hardswish: dst=work, src=dst (preserved), tmp=buf
        tile_std::kernel_ops::hardswish_f32(&mut work, &dst, &mut buf, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// mean + add + softmax + tanh + scaling
/// Maps to fuse/convtranspose3d_mean_add_softmax_tanh_scaling.py
/// reduce_mean → single f32 output
#[tile_std::tile_kernel]
pub fn convtranspose3d_mean_add_softmax_tanh_scaling(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // reduce mean → single f32
        let result = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);

        *output = result;
    }
}

/// relu + groupnorm
/// Maps to fuse/convtranspose3d_relu_groupnorm.py
/// relu + layernorm
#[tile_std::tile_kernel]
pub fn convtranspose3d_relu_groupnorm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut dst = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // relu
        tile_std::kernel_ops::relu_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // layernorm: dst != src
        tile_std::kernel_ops::layernorm_f32(&mut dst, &buf, &mut work, n, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, dst, n);
    }
}
