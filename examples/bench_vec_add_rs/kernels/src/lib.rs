#![feature(no_core)]
#![no_std]
#![no_core]

/// Tiled f16 vec_add benchmark kernel matching the C++ bench_vec_add_cpp interface.
///
/// Parameters match KernelVecAdd in vec_add_kernel.cpp:
///   x, y, z  — half-precision arrays (u16 in Rust)
///   len_buf  — pointer to per-block element count
///
/// Multi-block: each AICore block processes its own slice starting at
/// `get_block_idx() * n` (read from len_buf). Tiled in 256-element chunks.
///
/// Written against the safe `UbView<CAP, T>` Buffer API — the tile size
/// (`TILE`) is a const generic, so operand-shape mismatches between `bx`,
/// `by`, `bz` are compile errors.
use tile_std::buf::{
    ub_add_f16, ub_load_f16, ub_store_f16, UbCtx, UbView,
};

#[tile_std::tile_kernel]
pub fn vec_add_bench(x: *const u16, y: *const u16, z: *mut u16, len_buf: *const u32) {
    const TILE: usize = 256;
    unsafe {
        let n = *len_buf;
        let block_idx = tile_std::get_block_idx() as u32;
        let base_offset = block_idx * n;

        let ctx = UbCtx::new();
        let bz: UbView<'_, TILE, u16> = ctx.alloc::<TILE, u16>();

        let mut offset = 0u32;
        loop {
            if offset >= n {
                break;
            }
            let mut len = TILE as u32;
            if offset + len > n {
                len = n - offset;
            }
            let gm_off = (base_offset + offset) as usize;

            let bx = ub_load_f16::<TILE>(&ctx, x.wrapping_add(gm_off), len).sync();
            let by = ub_load_f16::<TILE>(&ctx, y.wrapping_add(gm_off), len).sync();

            ub_add_f16(&bz, &bx, &by, len);

            ub_store_f16(z.wrapping_add(gm_off), &bz, len);

            offset = offset + TILE as u32;
        }
    }
}

/// Vectorized f16 vec_add kernel using AscendC vector intrinsics.
///
/// Input layout: `x`, `y`, `z` are half-precision arrays, `len_buf` is a
/// uint32 pointer containing the per-block element count.
///
/// Uses multi-block distribution via get_block_idx/get_block_num.
/// Each block processes `n` elements starting at `block_idx * n`,
/// tiled into 256-element chunks to avoid UB overflow.
#[tile_std::tile_kernel]
pub fn vec_add(x: *const u16, y: *const u16, z: *mut u16, len_buf: *const u32) {
    const TILE: usize = 256;
    unsafe {
        let n = *len_buf;
        let block_idx = tile_std::get_block_idx() as u32;
        let base_offset = block_idx * n;

        let ctx = UbCtx::new();
        let bz: UbView<'_, TILE, u16> = ctx.alloc::<TILE, u16>();

        let mut offset = 0u32;
        loop {
            if offset >= n {
                break;
            }
            let mut len = TILE as u32;
            if offset + len > n {
                len = n - offset;
            }
            let gm_off = (base_offset + offset) as usize;

            // DMA load: GM -> UB (each returns DmaPending; .sync() inserts
            // the DMA→VEC barrier and produces a usable UbView).
            let bx = ub_load_f16::<TILE>(&ctx, x.wrapping_add(gm_off), len).sync();
            let by = ub_load_f16::<TILE>(&ctx, y.wrapping_add(gm_off), len).sync();

            // Vector add — all three operands must have CAP = TILE.
            ub_add_f16(&bz, &bx, &by, len);

            // DMA store: UB -> GM (auto VEC→DMA barrier).
            ub_store_f16(z.wrapping_add(gm_off), &bz, len);

            offset = offset + TILE as u32;
        }
    }
}
