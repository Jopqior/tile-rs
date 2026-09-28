// build-pass

// Blog Appendix C.4: Fused matmul+sigmoid kernel (tile-rs mitigation example)
// From blog/ascend-rs-memory-safe-npu-programming-en.md Appendix C.4

#![feature(no_core)]
#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn fused_matmul_sigmoid(
    a: *const u16, b: *const u16, c: *mut f32, dims: *const u32,
) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        // V6 mitigated: matmul_f16 handles DMA+cube internally
        tile_std::kernel_ops::matmul_f16(c, a, b, m, k, n);
        tile_std::__tile_pipe_barrier();  // Explicit, visible

        let total = m * n;
        let buf_c = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf_c, c as *const f32, total);
        tile_std::__tile_pipe_barrier();  // Explicit, visible

        // V6 mitigated: sigmoid_f32 includes ALL internal barriers
        // (muls → barrier → exp → barrier → adds → barrier → reciprocal)
        tile_std::kernel_ops::sigmoid_f32(buf_c, buf_c, total);

        tile_std::__tile_pipe_barrier();  // Explicit, visible
        tile_std::__tile_buf_store_f32(c, buf_c, total);
        // V4/V5: No FreeTensor — buf_c auto-managed
    }
}
