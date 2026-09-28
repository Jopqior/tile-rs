// build-pass

// Layer normalization kernel using composite helper.
// Normalizes input to zero mean and unit variance.

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn layernorm(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let eps = 1.0e-5f32;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf_in, &mut buf_work, n, eps);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
