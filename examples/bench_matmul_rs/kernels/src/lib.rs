#![feature(no_core)]
#![no_std]
#![no_core]

/// Fixed 32×32×32 matmul benchmark kernel matching bench_matmul_cpp interface.
///
/// Equivalent to C++ KernelMatmul (f16 × f16 → f32, m=n=k=32).
/// Uses kernel_ops::matmul_f16 which implements the full cube pipeline.
#[tile_std::tile_kernel]
pub fn matmul_bench(a: *const u16, b: *const u16, c: *mut f32) {
    unsafe {
        let m = 32u32;
        let k = 32u32;
        let n = 32u32;
        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
    }
}

/// Matrix multiplication kernel: C[m,n] = A[m,k] * B[k,n]
///
/// A, B are f16 (passed as *const u16), C is f32 (passed as *mut f32).
/// dims_buf contains [m, k, n] as u32.
///
/// Uses the tile_std matmul_f16 composite which handles the full
/// cube pipeline: GM -> L1 -> L0A/L0B -> Mmad -> L0C -> UB -> GM
#[tile_std::tile_kernel]
pub fn matmul(a: *const u16, b: *const u16, c: *mut f32, dims_buf: *const u32) {
    unsafe {
        let m = *dims_buf;
        let k = *dims_buf.wrapping_add(1);
        let n = *dims_buf.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
    }
}
