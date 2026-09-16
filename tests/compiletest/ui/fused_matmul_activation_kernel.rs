// build-pass

// Fused matmul + activation kernels.
// Maps to MultiKernelBench/reference/fuse/ category (matmul_* entries).

#![feature(no_core)]

#![no_std]
#![no_core]

/// matmul + GELU: C = gelu(A * B)
/// Maps to fuse/matmul_gelu_softmax.py (gelu part)
#[tile_std::tile_kernel]
pub fn fused_matmul_gelu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        // Matmul: C = A * B (result in UB via matmul_f16)
        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        // Load result and apply GELU
        let total = m * n;
        let buf_c = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut buf_tmp = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // gelu: dst=buf_out, src=buf_c (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf_out, &buf_c, &mut buf_tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}

/// matmul + ReLU: C = relu(A * B)
/// Maps to fuse/gemm_add_relu.py (relu part)
#[tile_std::tile_kernel]
pub fn fused_matmul_relu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf_c = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::relu_f32(buf_c, buf_c, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_c, total);
    }
}

/// matmul + sigmoid: C = sigmoid(A * B)
/// Maps to fuse/gemm_sigmoid_scaling_residual_add.py (sigmoid part)
#[tile_std::tile_kernel]
pub fn fused_matmul_sigmoid(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf_c = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::sigmoid_f32(buf_c, buf_c, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_c, total);
    }
}

/// matmul + tanh: C = tanh(A * B)
/// Maps to fuse/matmul_add_swish_tanh_gelu_hardtanh.py (tanh part)
#[tile_std::tile_kernel]
pub fn fused_matmul_tanh(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf_c = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::tanh_f32(buf_c, buf_c, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_c, total);
    }
}

/// matmul + swish: C = swish(A * B)
/// Maps to fuse/matmul_add_swish_tanh_gelu_hardtanh.py (swish part)
#[tile_std::tile_kernel]
pub fn fused_matmul_swish(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf_c = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut buf_tmp = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // swish: dst=buf_out, src=buf_c (preserved), tmp
        tile_std::kernel_ops::swish_f32(&mut buf_out, &buf_c, &mut buf_tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}

/// matmul + mish + mish: C = mish(mish(A * B))
/// Maps to fuse/matmul_mish_mish.py
#[tile_std::tile_kernel]
pub fn fused_matmul_mish_mish(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let mut buf_c = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut buf_tmp = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // mish: dst=buf_out, src=buf_c (preserved), tmp
        tile_std::kernel_ops::mish_f32(&mut buf_out, &buf_c, &mut buf_tmp, total);
        tile_std::__tile_pipe_barrier();
        // mish: dst=buf_c (dead), src=buf_out (preserved), tmp=buf_tmp
        tile_std::kernel_ops::mish_f32(&mut buf_c, &buf_out, &mut buf_tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_c, total);
    }
}

/// matmul + hardtanh: C = clamp(A * B, -1, 1)
/// Maps to fuse/matmul_add_swish_tanh_gelu_hardtanh.py (hardtanh part)
#[tile_std::tile_kernel]
pub fn fused_matmul_hardtanh(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf_c = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::hardtanh_f32(buf_c, buf_c, -1.0f32, 1.0f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_c, total);
    }
}

/// matmul + divide + gelu: C = gelu(A * B / scalar)
/// Maps to fuse/matmul_divide_gelu.py
#[tile_std::tile_kernel]
pub fn fused_matmul_divide_gelu(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        let divisor = 2.0f32;

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf_c = tile_std::__tile_buf_alloc(total);
        let mut buf_out = tile_std::__tile_buf_alloc(total);
        let mut buf_tmp = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // divide by scalar
        tile_std::__tile_muls_f32(buf_c, buf_c, 1.0f32 / divisor, total);
        tile_std::__tile_pipe_barrier();

        // gelu: dst=buf_out, src=buf_c (preserved), tmp
        tile_std::kernel_ops::gelu_f32(&mut buf_out, &buf_c, &mut buf_tmp, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_out, total);
    }
}

/// matmul + min + subtract: C = min(A*B, threshold) - bias
/// Maps to fuse/matmul_min_subtract.py
#[tile_std::tile_kernel]
pub fn fused_matmul_min_sub(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();

        let total = m * n;
        let buf_c = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();

        // min with threshold
        tile_std::__tile_mins_f32(buf_c, buf_c, 1.0f32, total);
        tile_std::__tile_pipe_barrier();
        // subtract bias
        tile_std::__tile_adds_f32(buf_c, buf_c, -0.5f32, total);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(c, buf_c, total);
    }
}
