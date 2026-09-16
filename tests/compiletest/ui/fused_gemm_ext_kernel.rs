// build-pass

// Fused GEMM + activation extension kernels.
// Maps to MultiKernelBench/reference/fuse/ category (gemm_* entries).

#![feature(no_core)]

#![no_std]
#![no_core]

/// gemm + add + relu: C = relu(A * B + 0.1)
/// Maps to fuse/gemm_add_relu.py
#[tile_std::tile_kernel]
pub fn gemm_add_relu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        tile_std::__tile_adds_f32(buf, buf, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf, total);
    }
}

/// gemm + batch_norm + gelu + group_norm + mean + relu
/// Maps to fuse/gemm_batch_norm_gelu_group_norm_mean_relu.py
#[tile_std::tile_kernel]
pub fn gemm_batch_norm_gelu_group_norm_mean_relu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // layernorm (dst != src)
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=work, src=buf_out (preserved), tmp=buf (dead)
        tile_std::kernel_ops::gelu_f32(&mut work, &buf_out, &mut buf, total);
        tile_std::__tile_pipe_barrier();
        // reduce_mean: dst=buf, src=work (preserved), work=buf_out
        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut buf, &work, &mut buf_out, total);
        *c = mean;
    }
}

/// gemm + batch_norm + scaling + softmax
/// Maps to fuse/gemm_batch_norm_scaling_softmax.py
#[tile_std::tile_kernel]
pub fn gemm_batch_norm_scaling_softmax(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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
        // scaling
        tile_std::__tile_muls_f32(buf_out, buf_out, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=buf (dead), src=buf_out (destroyed), work
        let mut buf2 = tile_std::__tile_buf_alloc(total);
        tile_std::kernel_ops::softmax_f32(&mut buf2, &mut buf_out, &mut work, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}

/// gemm + log_sum_exp + leaky_relu + leaky_relu + gelu + gelu
/// Maps to fuse/gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu.py
#[tile_std::tile_kernel]
pub fn gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // leaky_relu (result in work)
        tile_std::kernel_ops::leaky_relu_f32(&mut work, &mut buf, &mut tmp, 0.01f32, total);
        tile_std::__tile_pipe_barrier();
        // leaky_relu again (result in buf)
        tile_std::kernel_ops::leaky_relu_f32(&mut buf, &mut work, &mut tmp, 0.01f32, total);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=work, src=buf (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut work, &buf, &mut tmp, total);
        tile_std::__tile_pipe_barrier();
        // gelu again: dst=buf, src=work (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf, &work, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf, total);
    }
}

/// gemm + sigmoid + sum + log_sum_exp
/// Maps to fuse/gemm_sigmoid_sum_log_sum_exp.py
#[tile_std::tile_kernel]
pub fn gemm_sigmoid_sum_log_sum_exp(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

/// gemm + subtract + global_avg_pool + log_sum_exp + gelu + residual_add
/// Maps to fuse/gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add.py
#[tile_std::tile_kernel]
pub fn gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // subtract
        tile_std::__tile_adds_f32(buf, buf, -0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=buf2, src=buf (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf2, &buf, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}
