// build-pass

// Softmax kernel: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
// Full numerically-stable softmax using vector ops:
//   1. ReduceMax -> find max value
//   2. Adds(-max) -> subtract max for numerical stability
//   3. Exp -> exponentiate
//   4. ReduceSum -> sum of exponentials
//   5. Muls(1/sum) -> normalize

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn softmax(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // Step 1: find max(x) for numerical stability
        let max_val = tile_std::__tile_v_reduce_max_f32(buf_work, buf_in, buf_out, n);
        tile_std::__tile_pipe_barrier();

        // Step 2: buf_out = x - max(x)
        tile_std::__tile_adds_f32(buf_out, buf_in, -max_val, n);
        tile_std::__tile_pipe_barrier();

        // Step 3: buf_out = exp(x - max(x))
        tile_std::__tile_v_exp_f32(buf_out, buf_out, n);
        tile_std::__tile_pipe_barrier();

        // Save exp values into buf_in (no longer needed) before reduce corrupts buf_out
        tile_std::__tile_muls_f32(buf_in, buf_out, 1.0f32, n);
        tile_std::__tile_pipe_barrier();

        // Step 4: sum = sum(exp(x - max(x))) — buf_out may be corrupted, buf_in is safe
        let sum = tile_std::__tile_v_reduce_sum_f32(buf_work, buf_in, buf_out, n);
        tile_std::__tile_pipe_barrier();

        // Step 5: normalize from saved copy
        let inv_sum = 1.0f32 / sum;
        tile_std::__tile_muls_f32(buf_out, buf_in, inv_sum, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
