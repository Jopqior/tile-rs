#![feature(no_core)]
#![no_std]
#![no_core]

/// Tile matmul kernels: C[M×N] = A[M×K] @ B[K×N], f32.
///
/// Each kernel receives three flat global-memory pointers (A, B, C) and
/// computes a single tile's worth of matrix multiply.  The host selects
/// block_dim to equal the number of independent tiles.
///
/// Codegen note: mlir_to_cpp lowers `tile_matmul_f32` to a scalar triple
/// loop (correct, not fast) because the cube unit requires L0A/L0B/L0C
/// address spaces not accessible through TBuf<VECCALC>.  The mlir_to_pto
/// path (pto.tmatmul) targets the cube unit and is the intended fast path.
///
/// Written as shape-(B) `#[tile_kernel]` fns — each operand is a typed
/// `GmView` / `GmViewMut` at the ABI boundary. The macro synthesises the
/// boundary prelude (get_block_idx → GmDeviceCtx → per-operand
/// view{,_mut}::<R,C,T>), so the kernel body has zero `unsafe` blocks
/// and shape mismatches between A, B, and C are compile-time errors.
use tile_std::tile::{GmView, GmViewMut, tile_load_view_f32, tile_store_view_f32, safe};

macro_rules! tile_matmul_kernel {
    ($name:ident, $m:literal, $k:literal, $n:literal) => {
        #[tile_std::tile_kernel]
        pub fn $name(
            a: GmView<'_, $m, $k, f32>,
            b: GmView<'_, $k, $n, f32>,
            c: GmViewMut<'_, $m, $n, f32>,
        ) {
            let ta = tile_load_view_f32(&a);
            let tb = tile_load_view_f32(&b);
            let tc = safe::tile_matmul_f32(ta, tb);
            tile_store_view_f32(&c, tc);
        }
    };
}

// Small shapes that fit in UB (TBuf<VECCALC>) for the scalar fallback.
// M*K + K*N + M*N ≤ ~12288 f32 elements (48 KiB UB limit).
tile_matmul_kernel!(tile_matmul_16x16x16, 16, 16, 16);  //  16+16+16 = 48*4 = 768 B  ✓
tile_matmul_kernel!(tile_matmul_32x32x32, 32, 32, 32);  // 3*1024*4 = 12 KiB         ✓
tile_matmul_kernel!(tile_matmul_64x64x64, 64, 64, 64);  // 3*4096*4 = 48 KiB         ✓ (at limit)
tile_matmul_kernel!(tile_matmul_16x32x16, 16, 32, 16);  // 512+512+256 = 5 KiB       ✓
tile_matmul_kernel!(tile_matmul_32x64x32, 32, 64, 32);  // 2K+2K+1K = 20 KiB         ✓
