// Three 256-element f32 buffers, in pointer order a, b, out.
// The released rustc_codegen_tile backend compiles this source into MLIR on macOS.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::tile::{__tile_add_f32, __tile_load_f32, __tile_store_f32, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn add_generated(a: GmView<1, 256, f32>, b: GmView<1, 256, f32>, out: GmViewMut<1, 256, f32>) {
    unsafe {
        let a_tile = __tile_load_f32(a.as_ptr(), 1, 256);
        let b_tile = __tile_load_f32(b.as_ptr(), 1, 256);
        let result = __tile_add_f32(0, a_tile, b_tile, 1, 256);
        __tile_store_f32(out.as_mut_ptr(), result, 1, 256);
    }
}
