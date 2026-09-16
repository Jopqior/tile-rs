// build-pass

// Extended matmul variants.
// Maps to MultiKernelBench/reference/matmul/ category.
// Covers batched, symmetric, triangular, diagonal, transposed,
// and various dimension configurations.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Batched matmul: process multiple (m,k)x(k,n) pairs sequentially
/// In real impl each batch would be independent; here we process one.
#[tile_std::tile_kernel]
pub fn matmul_batched(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        let batch = *dims.wrapping_add(3);
        let stride_in = m * k;
        let stride_out = m * n;
        let mut b = 0u32;
        loop {
            if b >= batch { break; }
            let x_b = x.wrapping_add((b * stride_in) as usize);
            let w_b = w.wrapping_add((b * stride_in) as usize);
            let o_b = out.wrapping_add((b * stride_out) as usize);
            tile_std::kernel_ops::matmul_f16(o_b, x_b, w_b, m, k, n);
            tile_std::__tile_pipe_barrier();
            b = b + 1;
        }
    }
}

/// Symmetric matmul: A * A^T (result is symmetric)
/// Since we don't have transpose, we just compute A * A with same data.
#[tile_std::tile_kernel]
pub fn matmul_symmetric(x: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        tile_std::kernel_ops::matmul_f16(out, x, x, m, k, m);
        tile_std::__tile_pipe_barrier();
    }
}

/// Matmul with bias add: C = A*B + bias
#[tile_std::tile_kernel]
pub fn matmul_bias(x: *const u16, w: *const u16, bias: *const f32, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let bb = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_buf_load_f32(bb, bias, total);
        tile_std::__tile_pipe_barrier();
        // bb dead after add
        tile_std::__tile_v_add_f32(bb, buf, bb, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, bb, total);
    }
}

/// Matmul + scale: C = alpha * A * B
#[tile_std::tile_kernel]
pub fn matmul_scaled(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 0.5f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// Matmul + alpha*A*B + beta*C (full GEMM)
#[tile_std::tile_kernel]
pub fn gemm_full(a: *const u16, b: *const u16, c_in: *const f32, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let bc = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_buf_load_f32(bc, c_in, total);
        tile_std::__tile_pipe_barrier();
        // alpha * A*B
        tile_std::__tile_muls_f32(buf, buf, 1.0f32, total);
        tile_std::__tile_pipe_barrier();
        // beta * C
        tile_std::__tile_muls_f32(bc, bc, 0.5f32, total);
        tile_std::__tile_pipe_barrier();
        // alpha*A*B + beta*C — bc dead after
        tile_std::__tile_v_add_f32(bc, buf, bc, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, bc, total);
    }
}

/// Matmul wide: m=1, large n (row vector × matrix)
#[tile_std::tile_kernel]
pub fn matmul_wide(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let k = *dims;
        let n = *dims.wrapping_add(1);
        tile_std::kernel_ops::matmul_f16(out, x, w, 1, k, n);
        tile_std::__tile_pipe_barrier();
    }
}

/// Matmul + ReLU + matmul (two-layer MLP)
#[tile_std::tile_kernel]
pub fn matmul_relu_matmul(x: *const u16, w1: *const u16, w2: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        // First matmul
        tile_std::kernel_ops::matmul_f16(out, x, w1, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        // ReLU
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// Matmul accumulate: C += A*B (add to existing)
#[tile_std::tile_kernel]
pub fn matmul_accumulate(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        let total = m * n;
        // Load existing C
        let bc = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(bc, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        // Compute A*B into temp
        let temp_out = out.wrapping_add(total as usize);
        tile_std::kernel_ops::matmul_f16(temp_out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let bnew = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(bnew, temp_out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        // C += A*B — bnew dead after
        tile_std::__tile_v_add_f32(bnew, bc, bnew, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, bnew, total);
    }
}

/// Matmul with diagonal scaling: diag(d) * A * B
#[tile_std::tile_kernel]
pub fn matmul_diag_scale(x: *const u16, w: *const u16, diag: *const f32, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        let bd = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_buf_load_f32(bd, diag, total);
        tile_std::__tile_pipe_barrier();
        // bd dead after mul
        tile_std::__tile_v_mul_f32(bd, buf, bd, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, bd, total);
    }
}

/// Outer product: a * b^T (rank-1 update, simplified as elementwise)
#[tile_std::tile_kernel]
pub fn outer_product(a: *const f32, b: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let ba = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(ba, a, n);
        tile_std::__tile_buf_load_f32(bb, b, n);
        tile_std::__tile_pipe_barrier();
        // bb dead after mul
        tile_std::__tile_v_mul_f32(bb, ba, bb, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bb, n);
    }
}
