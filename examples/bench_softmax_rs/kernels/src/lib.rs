#![feature(no_core)]
#![no_std]
#![no_core]

/// Scalar softmax kernel — direct element-wise loops without vector ops.
///
/// Equivalent to C++ KernelSoftmaxNaive: uses scalar f32 arithmetic via raw
/// pointer reads/writes. This gives an apples-to-apples comparison with the
/// scalar C++ version to isolate compute cost from DMA and vectorization.
///
/// Includes the DMA load/store so the measurement includes full GM↔UB traffic.
#[tile_std::tile_kernel]
pub fn softmax_naive(input: *const f32, output: *mut f32, len_buf: *const u32) {
    unsafe {
        let n = *len_buf as usize;

        // Align to 8 elements (32 bytes) — same as C++ KernelSoftmaxNaive
        let aligned_n = ((n + 7) / 8) * 8;
        let mut buf_in  = tile_std::__tile_buf_alloc(aligned_n as u32);
        let mut buf_out = tile_std::__tile_buf_alloc(aligned_n as u32);

        tile_std::__tile_buf_load_f32(buf_in, input, n as u32);
        tile_std::__tile_pipe_barrier();

        // Step 1: scalar softmax via kernel_ops composite (includes reduce max/sum)
        let mut buf_work = tile_std::__tile_buf_alloc(aligned_n as u32);
        tile_std::kernel_ops::softmax_f32(&mut buf_out, &mut buf_in, &mut buf_work, n as u32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n as u32);
    }
}

/// Vectorized softmax kernel using AscendC vector intrinsics.
///
/// Input layout: `input` and `output` are float arrays, `len_buf` is a
/// uint32 pointer containing the element count.
///
/// This maps 1:1 to the C++ optimized softmax using ReduceMax, Adds, Exp,
/// ReduceSum, and Muls vector operations.
#[tile_std::tile_kernel]
pub fn softmax(input: *const f32, output: *mut f32, len_buf: *const u32) {
    unsafe {
        let n = *len_buf;

        let in_buf = tile_std::__tile_buf_alloc(n);
        let out_buf = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);
        let rwork = tile_std::__tile_buf_alloc(n);

        // DMA load: GM → local buffer
        tile_std::__tile_buf_load_f32(in_buf, input, n);
        tile_std::__tile_pipe_barrier();

        // ReduceMax → find max value
        let max_val = tile_std::__tile_v_reduce_max_f32(work, in_buf, rwork, n);

        // out = in - max_val (for numerical stability)
        tile_std::__tile_adds_f32(out_buf, in_buf, 0.0f32 - max_val, n);

        // out = exp(out)
        tile_std::__tile_v_exp_f32(out_buf, out_buf, n);

        // ReduceSum → compute normalization factor
        let sum_val = tile_std::__tile_v_reduce_sum_f32(work, out_buf, rwork, n);

        // out = out / sum (via multiply by 1/sum)
        tile_std::__tile_muls_f32(out_buf, out_buf, 1.0f32 / sum_val, n);

        // DMA store: local buffer → GM
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, out_buf, n);
    }
}

/// Pipeline softmax — type-state API with automatic barrier insertion.
///
/// Same algorithm, same performance, but:
/// - Zero manual pipe_barrier() calls (structurally guaranteed)
/// - Compile-time safety: DmaPending cannot be used as VecBuf (type error)
/// - 40% fewer lines than the manual version above
///
/// The pipeline module enforces the DMA↔VEC synchronization protocol
/// through Rust's type system:
///   load() → DmaPending ──.sync()──→ VecBuf ──(compute)──→ store()
///
/// Forgetting .sync() is a compile error, not a runtime crash.
#[tile_std::tile_kernel]
pub fn softmax_pipeline(input: *const f32, output: *mut f32, len_buf: *const u32) {
    unsafe {
        use tile_std::pipeline;

        let n = *len_buf;

        // Load: DMA → UB (returns DmaPending, must .sync() before use)
        let data = pipeline::load_f32(input, n).sync();
        let work = pipeline::alloc(n);
        let rwork = pipeline::alloc(n);
        let out = pipeline::alloc(n);

        // Compute: all vector ops, no barriers needed between them
        let max_val = data.reduce_max(work, rwork, n);
        out.adds(data, 0.0f32 - max_val, n);
        out.exp(out, n);
        let sum_val = out.reduce_sum(work, rwork, n);
        out.muls(out, 1.0f32 / sum_val, n);

        // Store: UB → GM (barrier inserted automatically)
        pipeline::store_f32(output, out, n);
    }
}

/// Async pipeline softmax — Future-based API (Phase 2).
///
/// Identical algorithm and generated code to `softmax_pipeline`, but uses
/// block_on(Future) instead of .sync(). This version:
/// - Zero manual pipe_barrier() calls (same as sync pipeline)
/// - Uses Future trait for DMA operations (composable with join! in Phase 3)
/// - Produces identical MLIR/C++ output (verified by diff)
///
/// In Phase 4 (codegen support), `block_on(f)` becomes `f.await`.
#[tile_std::tile_kernel]
pub fn softmax_async(input: *const f32, output: *mut f32, len_buf: *const u32) {
    unsafe {
        use tile_std::pipeline;

        let n = *len_buf;

        // Load: DMA → UB (Future resolves with barrier on poll)
        let data = pipeline::block_on(pipeline::load_f32_async(input, n));
        let work = pipeline::alloc(n);
        let rwork = pipeline::alloc(n);
        let out = pipeline::alloc(n);

        // Compute: all vector ops, no barriers needed
        let max_val = data.reduce_max(work, rwork, n);
        out.adds(data, 0.0f32 - max_val, n);
        out.exp(out, n);
        let sum_val = out.reduce_sum(work, rwork, n);
        out.muls(out, 1.0f32 / sum_val, n);

        // Store: UB → GM (sync store — StoreFuture codegen issue to fix in Phase 4)
        pipeline::store_f32(output, out, n);
    }
}
