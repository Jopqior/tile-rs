// build-pass

// Multi-operation fused kernels covering various combinations from
// MultiKernelBench/reference/fuse/ and other categories.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Instance norm + sum + residual add + multiply
/// Maps to fuse/bmm_instance_norm_sum_residual_add_multiply.py (partial)
#[tile_std::tile_kernel]
pub fn fused_norm_add_mul(x: *const f32, residual: *const f32, output: *mut f32, len: *const u32) {
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
        // residual add — br dead after
        tile_std::__tile_v_add_f32(br, tmp, br, n);
        tile_std::__tile_pipe_barrier();
        // multiply by 2
        tile_std::__tile_muls_f32(br, br, 2.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, br, n);
    }
}

/// Scale + batch_norm (simplified)
/// Maps to fuse/gemm_scale_batchnorm.py (partial)
#[tile_std::tile_kernel]
pub fn fused_scale_norm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(buf, buf, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// Subtract + mish + mish
/// Maps to fuse/conv2d_subtract_subtract_mish.py (partial)
#[tile_std::tile_kernel]
pub fn fused_sub_mish_mish(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut bx = tile_std::__tile_buf_alloc(n);
        let mut by = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        // by dead after sub (not used again)
        tile_std::__tile_v_sub_f32(tmp, bx, by, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::mish_f32(&mut bx, &tmp, &mut by, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::mish_f32(&mut tmp, &bx, &mut by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, tmp, n);
    }
}

/// Subtract + tanh + subtract + avg (partial avg = mean)
/// Maps to fuse/conv2d_subtract_tanh_subtract_avg_pool.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_sub_tanh_sub_mean(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        // first sub: bx - by → tmp (by still needed)
        tile_std::__tile_v_sub_f32(tmp, bx, by, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(tmp, tmp, n);
        tile_std::__tile_pipe_barrier();
        // second sub: tanh(x-y) - y → bx (by dead after)
        tile_std::__tile_v_sub_f32(bx, tmp, by, n);
        tile_std::__tile_pipe_barrier();

        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut tmp, &bx, &mut work, n);
        *output = mean;
    }
}

/// Min + add + multiply chain
/// Maps to fuse/conv2d_min_add_multiply.py (activation part)
#[tile_std::tile_kernel]
pub fn fused_min_add_mul(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_min_f32(tmp, bx, by, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bx, tmp, by, n);
        tile_std::__tile_pipe_barrier();
        // by dead after final mul
        tile_std::__tile_v_mul_f32(tmp, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, tmp, n);
    }
}

/// ELU + scaling chain
#[tile_std::tile_kernel]
pub fn fused_elu_scale(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::elu_f32(&mut work, &mut buf, &mut tmp, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(work, work, 2.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// SELU + add chain
#[tile_std::tile_kernel]
pub fn fused_selu_add(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        // selu destroys src(bx) and tmp — use work as dst
        tile_std::kernel_ops::selu_f32(&mut work, &mut bx, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // bx = selu(x) + y — all separate (bx != work != by)
        tile_std::__tile_v_add_f32(bx, work, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bx, n);
    }
}

/// Softplus + tanh (approximation of GELU variant)
#[tile_std::tile_kernel]
pub fn fused_softplus_tanh(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::softplus_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(buf, buf, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// ReLU + scale + add (residual connection after ReLU)
#[tile_std::tile_kernel]
pub fn fused_relu_scale_add(x: *const f32, residual: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let br = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(br, residual, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::relu_f32(bx, bx, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bx, bx, 0.5f32, n);
        tile_std::__tile_pipe_barrier();
        // br dead after add
        tile_std::__tile_v_add_f32(br, bx, br, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, br, n);
    }
}

/// Sigmoid + mul (gating mechanism)
#[tile_std::tile_kernel]
pub fn fused_sigmoid_gate(x: *const f32, gate: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bg, gate, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::sigmoid_f32(bg, bg, n);
        tile_std::__tile_pipe_barrier();
        // bg dead after
        tile_std::__tile_v_mul_f32(bg, bx, bg, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bg, n);
    }
}

/// Exp + reduce_sum (log-sum-exp denominator)
#[tile_std::tile_kernel]
pub fn fused_exp_reduce_sum(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_exp_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        let result = tile_std::__tile_v_reduce_sum_f32(work, buf, tmp, n);

        *output = result;
    }
}

/// Log-sum-exp: lse(x) = log(sum(exp(x)))
/// Maps to fuse/gemm_sigmoid_sum_log_sum_exp.py (partial)
#[tile_std::tile_kernel]
pub fn log_sum_exp(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // Numerically stable: lse(x) = max(x) + log(sum(exp(x - max(x))))
        let max_val = tile_std::__tile_v_reduce_max_f32(work, buf, tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, -max_val, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_exp_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        let sum = tile_std::__tile_v_reduce_sum_f32(work, buf, tmp, n);
        let result = max_val + tile_std::core::builtins::logf(sum);

        *output = result;
    }
}

/// Max + log + sum + exp (combined reduction)
/// Maps to fuse/conv3d_max_log_sum_exp_relu.py (partial)
#[tile_std::tile_kernel]
pub fn fused_max_lse_relu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // max
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        // log-sum-exp reduction
        let max_val = tile_std::__tile_v_reduce_max_f32(work, buf, tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, -max_val, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_exp_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        let sum = tile_std::__tile_v_reduce_sum_f32(work, buf, tmp, n);
        let result = max_val + tile_std::core::builtins::logf(sum);

        *output = result;
    }
}

/// Hardswish + mean + gelu (common in MobileNet fusions)
#[tile_std::tile_kernel]
pub fn fused_hardswish_gelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf2 = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // hardswish: dst=buf2, src=buf (preserved), tmp
        tile_std::kernel_ops::hardswish_f32(&mut buf2, &buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=tmp, src=buf2 (preserved), buf (dead)
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::kernel_ops::gelu_f32(&mut tmp, &buf2, &mut work, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, tmp, n);
    }
}

/// Softsign + scale + add
#[tile_std::tile_kernel]
pub fn fused_softsign_scale_add(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut ws = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        // softsign needs separate workspace to avoid src==workspace aliasing
        tile_std::kernel_ops::softsign_f32(&mut tmp, &bx, &mut ws, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(tmp, tmp, 2.0f32, n);
        tile_std::__tile_pipe_barrier();
        // by dead after add
        tile_std::__tile_v_add_f32(by, tmp, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, by, n);
    }
}

/// HardSigmoid + scale + clamp
#[tile_std::tile_kernel]
pub fn fused_hardsigmoid_scale_clamp(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::hardsigmoid_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 3.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::hardtanh_f32(buf, buf, 0.0f32, 2.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// Abs + sum (L1 loss variant)
#[tile_std::tile_kernel]
pub fn fused_abs_sum(x: *const f32, y: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        // by dead after sub
        tile_std::__tile_v_sub_f32(work, bx, by, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_abs_f32(work, work, n);
        tile_std::__tile_pipe_barrier();
        let result = tile_std::__tile_v_reduce_sum_f32(bx, work, by, n);

        *output = result / (n as f32);
    }
}

/// RMS norm + mish + scale
#[tile_std::tile_kernel]
pub fn fused_rmsnorm_mish_scale(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::rms_norm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // mish: dst=work, src=buf_out (preserved), tmp=buf (dead)
        let mut tmp = tile_std::__tile_buf_alloc(n);
        tile_std::kernel_ops::mish_f32(&mut work, &buf_out, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(work, work, 2.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// Reciprocal + scale + add (for 1/x normalization)
#[tile_std::tile_kernel]
pub fn fused_reciprocal_scale_add(x: *const f32, bias: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bb, bias, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_reciprocal_f32(bx, bx, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bx, bx, 0.5f32, n);
        tile_std::__tile_pipe_barrier();
        // bb dead after add
        tile_std::__tile_v_add_f32(bb, bx, bb, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bb, n);
    }
}
