// build-pass

// Double-buffered softmax as a K-tile loop marked with `pipelined_for(2)`.
//
// This is the canonical structural form the cpp emitter's `detect_tiling_loop`
// pass is built to recognise: a Rust `loop { … break … }` whose body contains
// at least one `__tile_buf_load_*` DMA, preceded by `pipelined_for(depth)`.
//
// Previously this file held a 2-tile manual unroll using
// `pipeline::join_dma_vec`; that pattern worked but did not expose a back-edge
// and so could not be driven by the marker. Rewriting it as a `loop` unblocks
// the pipelined_for(2) path end-to-end.

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn softmax_double_buf(input: *const f32, output: *mut f32, len_buf: *const u32) {
    unsafe {
        use tile_std::tile::pipelined_for;

        let n = *len_buf;
        let tile_size = n / 2u32;

        let mut buf = tile_std::__tile_buf_alloc(tile_size);
        let mut buf_out = tile_std::__tile_buf_alloc(tile_size);
        let mut work = tile_std::__tile_buf_alloc(tile_size);

        // Advise the backend to software-pipeline this K-loop with depth=2.
        // The cpp emitter pairs this with the back-edge and emits a
        // double-buffered TQue prologue/epilogue; PTO ignores the hint and
        // lets ptoas' `--enable-insert-sync` place cross-pipe syncs.
        pipelined_for(2);

        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::softmax_f32(&mut buf_out, &mut buf, &mut work, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf_out, len);
            offset = offset + tile_size;
        }
    }
}
