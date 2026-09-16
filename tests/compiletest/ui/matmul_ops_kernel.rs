// build-pass

// Matrix multiplication kernels using cube engine.
// Maps to MultiKernelBench/reference/matmul/ category.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Standard matrix multiplication: C = A * B
/// Maps to matmul/standard_matrix_multiplication.py
#[tile_std::tile_kernel]
pub fn matmul_standard(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
    }
}

/// Square matrix multiplication: C = A * B where A, B are NxN
/// Maps to matmul/square_matrix_multiplication.py
#[tile_std::tile_kernel]
pub fn matmul_square(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let n = *dims;
        tile_std::kernel_ops::matmul_f16(c, a, b, n, n, n);
    }
}

/// Matrix-vector multiplication: y = A * x where A is MxK, x is Kx1
/// Maps to matmul/matrix_vector_multiplication.py
#[tile_std::tile_kernel]
pub fn matmul_matvec(a: *const u16, x: *const u16, y: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        tile_std::kernel_ops::matmul_f16(y, a, x, m, k, 1);
    }
}

/// Matmul with large K dimension
/// Maps to matmul/matmul_with_large_k_dimension.py
#[tile_std::tile_kernel]
pub fn matmul_large_k(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
    }
}

/// Matmul with small K dimension
/// Maps to matmul/matmul_with_small_k_dimension.py
#[tile_std::tile_kernel]
pub fn matmul_small_k(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
    }
}

/// Matmul with irregular shapes
/// Maps to matmul/matmul_with_irregular_shapes.py
#[tile_std::tile_kernel]
pub fn matmul_irregular(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
    }
}

/// Tall-skinny matrix multiplication (M >> N)
/// Maps to matmul/tall_skinny_matrix_multiplication.py
#[tile_std::tile_kernel]
pub fn matmul_tall_skinny(a: *const u16, b: *const u16, c: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
    }
}
