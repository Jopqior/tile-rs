// build-pass

// Fused activation chain kernels — multi-step element-wise operations.
// These map to various entries in MultiKernelBench/reference/fuse/ that
// don't require convolution or matmul (pure vector activation chains).

#![feature(no_core)]

#![no_std]
#![no_core]

/// relu + hardswish chain
/// Maps to fuse/conv2d_relu_hard_swish.py (activation part only)
#[tile_std::tile_kernel]
pub fn fused_relu_hardswish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::relu_f32(buf_tmp, buf_in, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardswish_f32(&mut buf_out, &buf_tmp, &mut buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// hard_swish + relu chain
/// Maps to fuse/conv2d_hard_swish_relu.py (activation part only)
#[tile_std::tile_kernel]
pub fn fused_hardswish_relu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::hardswish_f32(&mut buf_out, &buf_in, &mut buf_tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf_out, buf_out, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// mish + mish chain
/// Maps to fuse/conv2d_mish_mish.py (activation part only)
#[tile_std::tile_kernel]
pub fn fused_mish_mish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::mish_f32(&mut buf_out, &buf_in, &mut buf_tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::mish_f32(&mut buf_tmp, &buf_out, &mut buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_tmp, n);
    }
}

/// mish + tanh chain
/// Maps to fuse/conv3d_mish_tanh.py (activation part only)
#[tile_std::tile_kernel]
pub fn fused_mish_tanh(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::mish_f32(&mut buf_out, &buf_in, &mut buf_tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(buf_out, buf_out, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// min + tanh + tanh chain
/// Maps to fuse/conv2d_min_tanh_tanh.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_min_tanh_tanh(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // min with threshold
        tile_std::__tile_mins_f32(buf_out, buf_in, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // tanh twice
        tile_std::kernel_ops::tanh_f32(buf_out, buf_out, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(buf_out, buf_out, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// multiply + leaky_relu + gelu chain
/// Maps to fuse/conv2d_multiply_leaky_relu_gelu.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_mul_leakyrelu_gelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // scale
        tile_std::__tile_muls_f32(buf_out, buf_in, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // leaky relu: result in buf_in, buf_out destroyed as src
        tile_std::kernel_ops::leaky_relu_f32(&mut buf_in, &mut buf_out, &mut buf_tmp, 0.01f32, n);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=buf_out, src=buf_in (preserved by gelu), tmp=buf_tmp
        tile_std::kernel_ops::gelu_f32(&mut buf_out, &buf_in, &mut buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// subtract + tanh + subtract chain
/// Maps to fuse/conv2d_subtract_subtract_mish.py (partial)
#[tile_std::tile_kernel]
pub fn fused_sub_tanh_sub(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        // subtract
        tile_std::__tile_v_sub_f32(bz, bx, by, n);
        tile_std::__tile_pipe_barrier();
        // tanh
        tile_std::kernel_ops::tanh_f32(bz, bz, n);
        tile_std::__tile_pipe_barrier();
        // subtract again
        tile_std::__tile_v_sub_f32(bz, bz, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bz, n);
    }
}

/// sigmoid + sum chain (element-wise sigmoid then reduce sum)
/// Maps to fuse/gemm_sigmoid_sum_log_sum_exp.py (partial)
#[tile_std::tile_kernel]
pub fn fused_sigmoid_sum(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // sigmoid
        tile_std::kernel_ops::sigmoid_f32(buf_in, buf_in, n);
        tile_std::__tile_pipe_barrier();
        // sum
        let result = tile_std::__tile_v_reduce_sum_f32(buf_work, buf_in, buf_tmp, n);

        *output = result;
    }
}

/// add + scale + sigmoid chain
/// Maps to fuse/conv2d_add_scale_sigmoid_group_norm.py (partial)
#[tile_std::tile_kernel]
pub fn fused_add_scale_sigmoid(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        // add — by dead after
        tile_std::__tile_v_add_f32(by, bx, by, n);
        tile_std::__tile_pipe_barrier();
        // scale
        tile_std::__tile_muls_f32(by, by, 0.5f32, n);
        tile_std::__tile_pipe_barrier();
        // sigmoid
        tile_std::kernel_ops::sigmoid_f32(by, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, by, n);
    }
}

/// scale + min chain
/// Maps to fuse/conv2d_scaling_min.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_scale_min(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(buf, buf, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_mins_f32(buf, buf, 1.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// leaky_relu + leaky_relu + gelu + gelu chain
/// Maps to fuse/gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu.py (partial)
#[tile_std::tile_kernel]
pub fn fused_leakyrelu_leakyrelu_gelu_gelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // leaky_relu chain: ping-pong buf↔work (src destroyed each call)
        tile_std::kernel_ops::leaky_relu_f32(&mut work, &mut buf, &mut tmp, 0.01f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::leaky_relu_f32(&mut buf, &mut work, &mut tmp, 0.01f32, n);
        tile_std::__tile_pipe_barrier();
        // gelu chain: ping-pong buf↔work (src preserved)
        tile_std::kernel_ops::gelu_f32(&mut work, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::gelu_f32(&mut buf, &work, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// divide + leaky_relu chain
/// Maps to fuse/conv2d_divide_leaky_relu.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_divide_leakyrelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(buf, buf, 0.5f32, n);
        tile_std::__tile_pipe_barrier();
        // leaky_relu: result in work, buf destroyed
        tile_std::kernel_ops::leaky_relu_f32(&mut work, &mut buf, &mut tmp, 0.01f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// subtract + hardswish chain
/// Maps to fuse/conv2d_subtract_hard_swish_max_pool_mish.py (partial)
#[tile_std::tile_kernel]
pub fn fused_sub_hardswish(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut bx = tile_std::__tile_buf_alloc(n);
        let mut by = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        // by dead after sub, reuse as workspace for hardswish
        tile_std::__tile_v_sub_f32(by, bx, by, n);
        tile_std::__tile_pipe_barrier();
        // hardswish: dst=tmp, src=by (preserved), work=bx
        tile_std::kernel_ops::hardswish_f32(&mut tmp, &by, &mut bx, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, tmp, n);
    }
}

/// tanh + scaling + bias_add + max chain
/// Maps to fuse/conv2d_tanh_scaling_bias_add_max.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_tanh_scale_bias_max(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        // tanh
        tile_std::kernel_ops::tanh_f32(bx, bx, n);
        tile_std::__tile_pipe_barrier();
        // scale
        tile_std::__tile_muls_f32(bx, bx, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // bias add — by dead after
        tile_std::__tile_v_add_f32(by, bx, by, n);
        tile_std::__tile_pipe_barrier();
        // max with 0
        tile_std::__tile_maxs_f32(by, by, 0.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, by, n);
    }
}

/// relu + bias_add chain
/// Maps to fuse/conv2d_relu_bias_add.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_relu_bias_add(x: *const f32, bias: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bb, bias, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::relu_f32(bx, bx, n);
        tile_std::__tile_pipe_barrier();
        // bb dead after add
        tile_std::__tile_v_add_f32(bb, bx, bb, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bb, n);
    }
}

/// hardswish + relu + softmax + mean chain
/// Maps to fuse/conv3d_hardswish_relu_softmax_mean.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_hardswish_relu_softmax_mean(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // hardswish: dst=work, src=buf (preserved), tmp
        tile_std::kernel_ops::hardswish_f32(&mut work, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, n);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=buf (dead), src=work (destroyed), tmp
        tile_std::kernel_ops::softmax_f32(&mut buf, &mut work, &mut tmp, n);
        tile_std::__tile_pipe_barrier();

        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);
        *output = mean;
    }
}

/// leaky_relu + sum + clamp + gelu chain
/// Maps to fuse/conv3d_leaky_relu_sum_clamp_gelu.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_leakyrelu_clamp_gelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // leaky_relu: result in work, buf destroyed as src
        tile_std::kernel_ops::leaky_relu_f32(&mut work, &mut buf, &mut tmp, 0.01f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardtanh_f32(work, work, -1.0f32, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=buf, src=work (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf, &work, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}
