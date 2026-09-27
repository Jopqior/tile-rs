// Three 256-element f32 buffers, in pointer order a, b, out.
// The released tile_kernel macro supports raw pointers without injecting GmView
// helper calls, which the current PTO emitter cannot preserve (run 36336996214).
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::tile::{__tile_add_f32, __tile_load_f32, __tile_store_f32};

#[tile_std::tile_kernel]
pub fn add_generated(a: *const f32, b: *const f32, out: *mut f32) {
    unsafe {
        let a_tile = __tile_load_f32(a, 1, 256);
        let b_tile = __tile_load_f32(b, 1, 256);
        let result = __tile_add_f32(0, a_tile, b_tile, 1, 256);
        __tile_store_f32(out, result, 1, 256);
    }
}
