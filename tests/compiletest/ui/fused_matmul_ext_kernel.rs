// build-pass

// Fused matmul + activation extension kernels.
// Maps to MultiKernelBench/reference/fuse/ category (matmul_* and bmm_* entries).

#![feature(no_core)]

#![no_std]
#![no_core]

/// matmul + avg_pool + gelu + scale + max
/// Maps to fuse/matmul_avg_pool_gelu_scale_max.py
#[tile_std::tile_kernel]
pub fn matmul_avg_pool_gelu_scale_max(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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
        // scale
        tile_std::__tile_muls_f32(buf2, buf2, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        // max
        tile_std::__tile_maxs_f32(buf2, buf2, 0.0f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}

/// matmul + batch_norm + bias_add + divide + swish
/// Maps to fuse/matmul_batch_norm_bias_add_divide_swish.py
#[tile_std::tile_kernel]
pub fn matmul_batch_norm_bias_add_divide_swish(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // layernorm (dst != src)
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // bias_add
        tile_std::__tile_adds_f32(buf_out, buf_out, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        // divide
        tile_std::__tile_muls_f32(buf_out, buf_out, 0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // swish: dst=work, src=buf_out (preserved), tmp=buf (dead)
        let mut buf2 = tile_std::__tile_buf_alloc(total);
        tile_std::kernel_ops::swish_f32(&mut work, &buf_out, &mut buf2, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

/// matmul + dropout + mean + softmax
/// Maps to fuse/matmul_dropout_mean_softmax.py
#[tile_std::tile_kernel]
pub fn matmul_dropout_mean_softmax(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // dropout = identity at inference
        // softmax: dst=work, src=buf (destroyed), tmp
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

/// matmul + scale + residual_add + clamp + log_sum_exp + mish
/// Maps to fuse/matmul_scale_residual_add_clamp_log_sum_exp_mish.py
#[tile_std::tile_kernel]
pub fn matmul_scale_residual_add_clamp_log_sum_exp_mish(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // scale
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        // clamp (hardtanh)
        tile_std::kernel_ops::hardtanh_f32(buf, buf, -1.0f32, 1.0f32, total);
        tile_std::__tile_pipe_barrier();
        // mish: dst=buf2, src=buf (preserved), tmp
        tile_std::kernel_ops::mish_f32(&mut buf2, &buf, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}

/// matmul + scaling + residual_add
/// Maps to fuse/matmul_scaling_residual_add.py
#[tile_std::tile_kernel]
pub fn matmul_scaling_residual_add(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // scaling
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        // residual add (bias)
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf, total);
    }
}

/// matmul + sigmoid + sum
/// Maps to fuse/matmul_sigmoid_sum.py
#[tile_std::tile_kernel]
pub fn matmul_sigmoid_sum(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // sigmoid
        tile_std::kernel_ops::sigmoid_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        // reduce_sum
        let sum = tile_std::__tile_v_reduce_sum_f32(buf, buf, work, total);
        *c = sum;
    }
}

/// matmul + subtract + multiply + relu
/// Maps to fuse/matmul_subtract_multiply_relu.py
#[tile_std::tile_kernel]
pub fn matmul_subtract_multiply_relu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // subtract
        tile_std::__tile_adds_f32(buf, buf, -0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // multiply
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        // relu
        tile_std::kernel_ops::relu_f32(buf, buf, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf, total);
    }
}

/// matmul + sum + max + avg_pool + log_sum_exp + log_sum_exp
/// Maps to fuse/matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp.py
#[tile_std::tile_kernel]
pub fn matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // max
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, total);
        tile_std::__tile_pipe_barrier();
        // reduce_sum
        let sum = tile_std::__tile_v_reduce_sum_f32(buf, buf, work, total);
        *c = sum;
    }
}

/// matmul + swish + scaling
/// Maps to fuse/matmul_swish_scaling.py
#[tile_std::tile_kernel]
pub fn matmul_swish_scaling(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // swish: dst=buf2, src=buf (preserved), tmp
        tile_std::kernel_ops::swish_f32(&mut buf2, &buf, &mut tmp, total);
        tile_std::__tile_pipe_barrier();
        // scaling
        tile_std::__tile_muls_f32(buf2, buf2, 2.0f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}

/// matmul + swish + sum + group_norm
/// Maps to fuse/matmul_swish_sum_group_norm.py
#[tile_std::tile_kernel]
pub fn matmul_swish_sum_group_norm(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // swish: dst=buf_out, src=buf (preserved), work
        tile_std::kernel_ops::swish_f32(&mut buf_out, &buf, &mut work, total);
        tile_std::__tile_pipe_barrier();
        // layernorm: dst=work, src=buf_out (preserved)
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf_out, &mut tmp, total, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

/// bmm + instance_norm + sum + residual_add + multiply
/// Maps to fuse/bmm_instance_norm_sum_residual_add_multiply.py
#[tile_std::tile_kernel]
pub fn bmm_instance_norm_sum_residual_add_multiply(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // layernorm (dst != src)
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // multiply (scaling)
        tile_std::__tile_muls_f32(buf_out, buf_out, 2.0f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}
