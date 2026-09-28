// build-pass

// Softmax kernel using pipeline type-state API.
// Zero manual pipe_barrier() calls — barriers are structurally guaranteed.

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn softmax_pipeline(input: *const f32, output: *mut f32, len_buf: *const u32) {
    unsafe {
        use tile_std::pipeline;

        let n = *len_buf;

        let data = pipeline::load_f32(input, n).sync();
        let work = pipeline::alloc(n);
        let rwork = pipeline::alloc(n);
        let out = pipeline::alloc(n);

        let max_val = data.reduce_max(work, rwork, n);
        out.adds(data, 0.0f32 - max_val, n);
        out.exp(out, n);
        let sum_val = out.reduce_sum(work, rwork, n);
        out.muls(out, 1.0f32 / sum_val, n);

        pipeline::store_f32(output, out, n);
    }
}
