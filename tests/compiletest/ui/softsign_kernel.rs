// build-pass

// Softsign activation kernel: softsign(x) = x / (1 + |x|)
// Maps to MultiKernelBench/reference/activation/softsign.py

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn softsign(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // softsign(x) = x / (1 + |x|) — 3-buffer to avoid dst aliasing in Mul
        // buf_tmp = |x|
        tile_std::__tile_v_abs_f32(buf_tmp, buf_in, n);
        tile_std::__tile_pipe_barrier();
        // buf_tmp = 1 + |x|
        tile_std::__tile_adds_f32(buf_tmp, buf_tmp, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // buf_tmp = 1 / (1 + |x|)
        tile_std::__tile_reciprocal_f32(buf_tmp, buf_tmp, n);
        tile_std::__tile_pipe_barrier();
        // buf_out = x * (1 / (1 + |x|))
        tile_std::__tile_v_mul_f32(buf_out, buf_in, buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
