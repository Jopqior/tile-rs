// build-pass

// Matrix multiply kernel using the cube engine (Mmad).
// C[m,n] = A[m,k] * B[k,n]  (A,B: f16, C: f32)
//
// Uses the high-level matmul_f16 composite which handles all
// data movement through the cube pipeline:
//   GM → L1 → L0A/L0B → Mmad → L0C → UB → GM

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn matmul(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    dims: *const u32,
) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
    }
}
