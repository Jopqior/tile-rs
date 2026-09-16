// build-pass

// Sigmoid activation kernel: sigmoid(x) = 1 / (1 + exp(-x))
// Composed from: Muls(-1) -> Exp -> Adds(1) -> Reciprocal
// Each step requires pipe_barrier(PIPE_ALL) on 310P for in-place chaining.

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn sigmoid(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::sigmoid_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
