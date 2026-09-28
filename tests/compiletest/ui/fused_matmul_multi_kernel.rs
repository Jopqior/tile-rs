// build-pass

// Fused matmul + multiple post-processing steps.
// Maps to deeper fuse/ entries and attention/ patterns.

#![feature(no_core)]

#![no_std]
#![no_core]

/// matmul + softmax (attention scores computation)
/// Maps to fuse/matmul_gelu_softmax.py (softmax part)
#[tile_std::tile_kernel]
pub fn fused_matmul_softmax(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // softmax: dst=work, src=buf (destroyed), tmp
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

/// matmul + gelu + softmax
/// Maps to fuse/matmul_gelu_softmax.py
#[tile_std::tile_kernel]
pub fn fused_matmul_gelu_softmax(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // gelu: dst=work, src=buf (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut work, &buf, &mut tmp, total);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=tmp, src=work (destroyed), buf (dead)
        let mut buf2 = tile_std::__tile_buf_alloc(total);
        tile_std::kernel_ops::softmax_f32(&mut buf2, &mut work, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}

/// matmul + layernorm
/// Maps to fuse/matmul_group_norm_leaky_relu_sum.py (norm part)
#[tile_std::tile_kernel]
pub fn fused_matmul_layernorm(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}

/// matmul + norm + leaky_relu + sum
/// Maps to fuse/matmul_group_norm_leaky_relu_sum.py
#[tile_std::tile_kernel]
pub fn fused_matmul_norm_leakyrelu_sum(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // leaky_relu: result in buf, buf_out destroyed as src
        tile_std::kernel_ops::leaky_relu_f32(&mut buf, &mut buf_out, &mut work, 0.01f32, total);
        tile_std::__tile_pipe_barrier();
        let sum = tile_std::__tile_v_reduce_sum_f32(buf, buf, work, total);

        *c = sum;
    }
}

/// matmul + batch_norm + bias + divide + swish
/// Maps to fuse/matmul_batch_norm_bias_add_divide_swish.py
#[tile_std::tile_kernel]
pub fn fused_matmul_norm_bias_div_swish(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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
        // bias
        tile_std::__tile_adds_f32(buf_out, buf_out, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        // divide
        tile_std::__tile_muls_f32(buf_out, buf_out, 0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // swish: dst=work, src=buf_out (preserved), tmp=buf (dead)
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::kernel_ops::swish_f32(&mut work, &buf_out, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

/// matmul + mean + softmax (attention with mean pooling)
/// Maps to fuse/matmul_dropout_mean_softmax.py
#[tile_std::tile_kernel]
pub fn fused_matmul_mean_softmax(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // softmax: dst=work, src=buf (destroyed), tmp
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

/// matmul + max + scale (max pooling + scaling after matmul)
/// Maps to fuse/matmul_max_pool_sum_scale.py (partial)
#[tile_std::tile_kernel]
pub fn fused_matmul_max_scale(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // clamp to max
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, total);
        tile_std::__tile_pipe_barrier();
        // sum
        let sum = tile_std::__tile_v_reduce_sum_f32(buf, buf, work, total);
        // scale
        *c = sum * 2.0f32;
    }
}

/// matmul + add + swish + tanh + gelu + hardtanh
/// Maps to fuse/matmul_add_swish_tanh_gelu_hardtanh.py
#[tile_std::tile_kernel]
pub fn fused_matmul_add_swish_tanh_gelu_hardtanh(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // add bias
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        // swish: dst=buf2, src=buf (preserved), tmp
        tile_std::kernel_ops::swish_f32(&mut buf2, &buf, &mut tmp, total);
        tile_std::__tile_pipe_barrier();
        // tanh
        tile_std::kernel_ops::tanh_f32(buf2, buf2, total);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=tmp, src=buf2 (preserved), buf (dead)
        let mut buf3 = tile_std::__tile_buf_alloc(total);
        tile_std::kernel_ops::gelu_f32(&mut tmp, &buf2, &mut buf3, total);
        tile_std::__tile_pipe_barrier();
        // hardtanh
        tile_std::kernel_ops::hardtanh_f32(tmp, tmp, -1.0f32, 1.0f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, tmp, total);
    }
}

/// matmul + subtract + global avg (reduce_mean)
/// Maps to fuse/gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add.py (partial)
#[tile_std::tile_kernel]
pub fn fused_matmul_sub_mean_gelu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // subtract bias
        tile_std::__tile_adds_f32(buf, buf, -0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=buf2, src=buf (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf2, &buf, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}
