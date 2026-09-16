// build-pass

// Tanh activation kernel: tanh(x) = 2 * sigmoid(2x) - 1
// Composed from: Muls(2) -> Muls(-1) -> Exp -> Adds(1) -> Reciprocal -> Muls(2) -> Adds(-1)

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn tanh_kernel(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::tanh_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
