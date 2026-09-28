// build-pass

// Blog Appendix D.4: Triton IR → tile-rs vector_add kernel
// From blog/ascend-rs-memory-safe-npu-programming-en.md Appendix D.4

#![feature(no_core)]
#![no_std]
#![no_core]

// Rust kernel: Triton IR → tile-rs instead of raw C++
#[tile_std::tile_kernel]  // ← triggers automatic __aicore__ in codegen
pub fn vector_add(
    x: *const f32, y: *const f32, z: *mut f32, n_ptr: *const u32,
) {
    unsafe {
        let n = *n_ptr;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_add_f32(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(z, bz, n);
    }
}
