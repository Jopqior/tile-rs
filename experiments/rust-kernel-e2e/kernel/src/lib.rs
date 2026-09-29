#![feature(no_core)]
#![no_std]
#![no_core]

use tile_std::tile::{tile_add_f32, tile_load_f32, tile_store_f32};

// One 1x8 f32 tile: c[i] = a[i] + b[i]. No GPU run in this experiment.
#[tile_std::tile_kernel]
pub fn add_f32_small(a: *const f32, b: *const f32, c: *mut f32) {
    let x = tile_load_f32::<1, 8>(a);
    let y = tile_load_f32::<1, 8>(b);
    let sum = tile_add_f32(x, y);
    tile_store_f32(c, sum);
}
