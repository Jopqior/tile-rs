// build-pass

// Blog Appendix D.2: TileLang DSL → tile-rs softmax kernel
// From blog/ascend-rs-memory-safe-npu-programming-en.md Appendix D.2

#![feature(no_core)]
#![no_std]
#![no_core]

// Rust kernel: TileLang DSL → tile-rs instead of raw C++
#[tile_std::tile_kernel]
pub fn tilelang_softmax(input: *const f32, output: *mut f32, n_ptr: *const u32) {
    unsafe {
        let n = *n_ptr;
        let mut buf_in  = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work    = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();  // codegen also auto-inserts after DMA

        // kernel_ops::softmax_f32 has 4 embedded pipe_barrier() calls —
        // impossible to forget any of them
        tile_std::kernel_ops::softmax_f32(&mut buf_out, &mut buf_in, &mut work, n);

        tile_std::__tile_pipe_barrier();  // codegen also auto-inserts before DMA
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
