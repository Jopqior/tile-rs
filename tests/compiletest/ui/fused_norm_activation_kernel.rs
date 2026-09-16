// build-pass

// Fused normalization + activation kernels.
// Maps to various fuse/ entries combining normalization with activations.

#![feature(no_core)]

#![no_std]
#![no_core]

/// layernorm + relu
/// Maps to fuse/gemm_batch_norm_gelu_group_norm_mean_relu.py (partial)
#[tile_std::tile_kernel]
pub fn fused_layernorm_relu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf_out, buf_out, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// layernorm + sigmoid
#[tile_std::tile_kernel]
pub fn fused_layernorm_sigmoid(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(buf_out, buf_out, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// rms_norm + swish
#[tile_std::tile_kernel]
pub fn fused_rmsnorm_swish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::rms_norm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // swish: dst=work, src=buf_out (preserved), tmp
        tile_std::kernel_ops::swish_f32(&mut work, &buf_out, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// layernorm + tanh + hardswish
#[tile_std::tile_kernel]
pub fn fused_layernorm_tanh_hardswish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(buf_out, buf_out, n);
        tile_std::__tile_pipe_barrier();
        // hardswish: dst=work, src=buf_out (preserved), tmp
        tile_std::kernel_ops::hardswish_f32(&mut work, &buf_out, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// softmax + mean (softmax followed by mean reduction)
/// Maps to fuse/matmul_dropout_mean_softmax.py (partial)
#[tile_std::tile_kernel]
pub fn fused_softmax_mean(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // softmax: dst=work, src=buf (destroyed), tmp
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut tmp, n);
        tile_std::__tile_pipe_barrier();
        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut buf, &work, &mut tmp, n);

        *output = mean;
    }
}

/// layernorm + gelu (common transformer building block)
#[tile_std::tile_kernel]
pub fn fused_layernorm_gelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=work, src=buf_out (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut work, &buf_out, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// rms_norm + gelu
#[tile_std::tile_kernel]
pub fn fused_rmsnorm_gelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::rms_norm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=work, src=buf_out (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut work, &buf_out, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// log_softmax + mean (for cross-entropy style losses)
#[tile_std::tile_kernel]
pub fn fused_log_softmax_mean(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work2 = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // Use separate dst/src to avoid aliasing: log_softmax's reduce_max(work, src, dst) destroys src when dst==src
        tile_std::kernel_ops::log_softmax_f32(&mut work, &mut buf, &mut tmp, &mut work2, n);
        tile_std::__tile_pipe_barrier();
        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut buf, &work, &mut tmp, n);

        *output = mean;
    }
}
