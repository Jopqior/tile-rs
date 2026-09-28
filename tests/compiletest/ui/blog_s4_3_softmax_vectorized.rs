// build-pass

// Blog section 4.3: Vectorized softmax kernel
// From blog/ascend-rs-memory-safe-npu-programming-en.md section 4.3
// (Blog shows snippet without headers; full compilable version here)

#![feature(no_core)]
#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn softmax(input: *const f32, output: *mut f32, len_buf: *const u32) {
    unsafe {
        let n = *len_buf;
        let in_buf  = tile_std::__tile_buf_alloc(n);
        let out_buf = tile_std::__tile_buf_alloc(n);
        let work    = tile_std::__tile_buf_alloc(n);
        let rwork   = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(in_buf, input, n);
        tile_std::__tile_pipe_barrier();

        let max_val = tile_std::__tile_v_reduce_max_f32(work, in_buf, rwork, n);
        tile_std::__tile_adds_f32(out_buf, in_buf, 0.0f32 - max_val, n);
        tile_std::__tile_v_exp_f32(out_buf, out_buf, n);
        let sum_val = tile_std::__tile_v_reduce_sum_f32(work, out_buf, rwork, n);
        tile_std::__tile_muls_f32(out_buf, out_buf, 1.0f32 / sum_val, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, out_buf, n);
    }
}
