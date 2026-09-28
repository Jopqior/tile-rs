// Case 4: Missing Sync — Prevented by Built-In Barriers in Composites
//
// In tile-rs, the kernel_ops module provides composite operations
// (sigmoid, softmax, gelu, etc.) that include pipe_barrier() calls
// between every pipeline-crossing step internally. The kernel author
// only needs to place barriers at the DMA/compute boundary — and
// this pattern is consistent and visible in every kernel.
//
// For the DMA boundary itself, the explicit __tile_pipe_barrier() call
// is required and clearly documents the synchronization point. Unlike
// the C++ queue model where synchronization semantics are implicit and
// easy to get wrong, the Rust API makes every barrier explicit.

#![feature(no_core)]
#![no_std]
#![no_core]

/// Sigmoid kernel with correct synchronization.
///
/// The pipe_barrier() between DMA load and compute is explicit and
/// visible. The sigmoid_f32 composite includes all internal barriers
/// between its four steps (muls → exp → adds → reciprocal).
///
/// Compare with the C++ version where the EnQue/DeQue queue model
/// provides the illusion of synchronization but does not actually
/// ensure DMA completion — the programmer must remember to add
/// pipe_barrier() or use the correct queue depth, and forgetting
/// produces silently wrong results.
#[tile_std::tile_kernel]
pub fn sigmoid(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        // DMA load from GM to UB
        tile_std::__tile_buf_load_f32(buf_in, input, n);

        // Explicit barrier: guarantees DMA load is complete before
        // any vector operations read from buf_in.
        tile_std::__tile_pipe_barrier();

        // sigmoid_f32 is a composite that internally does:
        //   muls(-1) → pipe_barrier → exp → pipe_barrier →
        //   adds(1) → pipe_barrier → reciprocal
        // All internal barriers are included — no way to forget one.
        tile_std::kernel_ops::sigmoid_f32(buf_out, buf_in, n);

        // Explicit barrier: guarantees vector compute is complete
        // before DMA store reads from buf_out.
        tile_std::__tile_pipe_barrier();

        // DMA store from UB to GM
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
