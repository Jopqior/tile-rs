#![feature(no_core)]
#![no_std]
#![no_core]

/// Tile softmax kernels for benchmarking.
///
/// 6 monomorphizations covering 1D (single-row) and multi-row shapes.
/// Each kernel loads a tile, applies row-wise softmax, stores the result.
///
/// Written against the shape-(B) safe GmView API — the `#[tile_kernel]`
/// attribute auto-synthesises the boundary prelude (get_block_idx →
/// GmDeviceCtx → per-view view{,_mut}) from the view-typed params, so
/// the kernel body contains zero `unsafe` blocks.
use tile_std::tile::{GmView, GmViewMut, tile_load_view_f32, tile_store_view_f32, safe};

macro_rules! tile_softmax_kernel {
    ($name:ident, $rows:literal, $cols:literal) => {
        #[tile_std::tile_kernel]
        pub fn $name(
            input:  GmView<'_, $rows, $cols, f32>,
            output: GmViewMut<'_, $rows, $cols, f32>,
        ) {
            let t = tile_load_view_f32(&input);
            let r = safe::tile_softmax_f32(t);
            tile_store_view_f32(&output, r);
        }
    };
}

tile_softmax_kernel!(tile_softmax_1x1024, 1, 1024);
tile_softmax_kernel!(tile_softmax_1x4096, 1, 4096);
tile_softmax_kernel!(tile_softmax_1x8192, 1, 8192);
tile_softmax_kernel!(tile_softmax_4x256, 4, 256);
tile_softmax_kernel!(tile_softmax_16x256, 16, 256);
tile_softmax_kernel!(tile_softmax_16x512, 16, 512);
