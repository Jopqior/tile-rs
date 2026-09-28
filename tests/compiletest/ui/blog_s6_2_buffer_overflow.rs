// build-pass

// Blog section 6.2: Buffer overflow case study (safe Rust version)
// Exact copy from blog/ascend-rs-memory-safe-npu-programming-en.md section 6.2

#![feature(no_core)]
#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn softmax(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::softmax_f32(&mut buf_out, &mut buf_in, &mut buf_work, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
