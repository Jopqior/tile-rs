// build-pass

// Tests SELU and Swish activation kernels using composite helpers.

#![feature(no_core)]

#![no_std]
#![no_core]

// --- SELU using composite helper ---
#[tile_std::tile_kernel]
pub fn test_selu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::selu_f32(&mut buf_out, &mut buf_in, &mut buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

// --- Swish using composite helper ---
#[tile_std::tile_kernel]
pub fn test_swish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::swish_f32(&mut buf_out, &buf_in, &mut buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
