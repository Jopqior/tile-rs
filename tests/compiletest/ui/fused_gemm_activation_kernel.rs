// build-pass

// Fused GEMM + activation kernels.
// Maps to MultiKernelBench/reference/fuse/ category (gemm_* entries).

#![feature(no_core)]

#![no_std]
#![no_core]

/// gemm + relu: C = relu(A * B)
/// Maps to fuse/gemm_add_relu.py
#[tile_std::tile_kernel]
pub fn fused_gemm_relu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        tile_std::kernel_ops::relu_f32(buf, buf, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf, total);
    }
}

/// gemm + relu + divide: C = relu(A * B) / scalar
/// Maps to fuse/gemm_relu_divide.py
#[tile_std::tile_kernel]
pub fn fused_gemm_relu_divide(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 0.5f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf, total);
    }
}

/// gemm + multiply + leakyrelu: C = leaky_relu(scale * A * B)
/// Maps to fuse/gemm_multiply_leakyrelu.py
#[tile_std::tile_kernel]
pub fn fused_gemm_multiply_leakyrelu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // scale
        tile_std::__tile_muls_f32(buf, buf, 2.0f32, total);
        tile_std::__tile_pipe_barrier();
        // leaky relu (result in work, buf destroyed)
        tile_std::kernel_ops::leaky_relu_f32(&mut work, &mut buf, &mut tmp, 0.01f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, work, total);
    }
}

/// gemm + sigmoid + scaling + residual add
/// Maps to fuse/gemm_sigmoid_scaling_residual_add.py
#[tile_std::tile_kernel]
pub fn fused_gemm_sigmoid_scale_add(
    a: *const u16, b: *const u16, residual: *const f32,
    c: *mut f32, dims: *const u32
) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let buf_res = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_buf_load_f32(buf_res, residual, total);
        tile_std::__tile_pipe_barrier();

        // sigmoid
        tile_std::kernel_ops::sigmoid_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        // scale
        tile_std::__tile_muls_f32(buf, buf, 0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // add residual — buf_res dead after
        tile_std::__tile_v_add_f32(buf_res, buf, buf_res, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_res, total);
    }
}

/// gemm + max + subtract + gelu
/// Maps to fuse/gemm_max_subtract_gelu.py
#[tile_std::tile_kernel]
pub fn fused_gemm_max_sub_gelu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf2 = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // max with 0
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, total);
        tile_std::__tile_pipe_barrier();
        // subtract constant
        tile_std::__tile_adds_f32(buf, buf, -0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=buf2, src=buf (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf2, &buf, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}

/// gemm + scaling + hardtanh + gelu
/// Maps to fuse/gemm_scaling_hard_tanh_gelu.py
#[tile_std::tile_kernel]
pub fn fused_gemm_scale_hardtanh_gelu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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
        // hardtanh
        tile_std::kernel_ops::hardtanh_f32(buf, buf, -1.0f32, 1.0f32, total);
        tile_std::__tile_pipe_barrier();
        // gelu: dst=buf2, src=buf (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf2, &buf, &mut tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}

/// gemm + divide + sum + scaling
/// Maps to fuse/gemm_divide_sum_scaling.py
#[tile_std::tile_kernel]
pub fn fused_gemm_divide_sum_scale(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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

        // divide
        tile_std::__tile_muls_f32(buf, buf, 0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // sum reduction
        let sum = tile_std::__tile_v_reduce_sum_f32(buf, buf, work, total);
        // scale
        let result = sum * 2.0f32;
        *c = result;
    }
}

/// gemm + swish + divide + clamp + tanh + clamp
/// Maps to fuse/gemm_swish_divide_clamp_tanh_clamp.py
#[tile_std::tile_kernel]
pub fn fused_gemm_swish_div_clamp_tanh_clamp(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
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
        // divide
        tile_std::__tile_muls_f32(buf2, buf2, 0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // clamp
        tile_std::kernel_ops::hardtanh_f32(buf2, buf2, -1.0f32, 1.0f32, total);
        tile_std::__tile_pipe_barrier();
        // tanh
        tile_std::kernel_ops::tanh_f32(buf2, buf2, total);
        tile_std::__tile_pipe_barrier();
        // clamp again
        tile_std::kernel_ops::hardtanh_f32(buf2, buf2, -0.5f32, 0.5f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf2, total);
    }
}
