// build-pass

// HardSigmoid activation kernel: hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)
// Maps to MultiKernelBench/reference/activation/hardsigmoid.py

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn hardsigmoid(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::hardsigmoid_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
