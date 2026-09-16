// build-pass

// Blog Appendix D.3: PyTorch torch.compile → tile-rs fused GELU kernel
// From blog/ascend-rs-memory-safe-npu-programming-en.md Appendix D.3

#![feature(no_core)]
#![no_std]
#![no_core]

// Rust kernel: torch.compile → tile-rs instead of raw C++
#[tile_std::tile_kernel]
pub fn fused_gelu(input: *const f32, output: *mut f32, n_ptr: *const u32) {
    unsafe {
        let n = *n_ptr;
        // Typed buffer IDs (UbBuf) — no pointer arithmetic, no sizing errors
        let buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // GELU via composites: x * sigmoid(1.702 * x)
        tile_std::kernel_ops::gelu_f32(&mut tmp, &buf, &mut work, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, tmp, n);
    }
}
