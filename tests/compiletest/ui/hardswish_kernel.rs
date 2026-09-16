// build-pass

// HardSwish activation kernel: hardswish(x) = x * hardsigmoid(x)
// Maps to fused conv2d_hard_swish operations in MultiKernelBench

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn hardswish(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // hardswish(x) = x * hardsigmoid(x) — 3-buffer to avoid dst aliasing in Mul
        // buf_tmp = hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)
        tile_std::kernel_ops::hardsigmoid_f32(buf_tmp, buf_in, n);
        tile_std::__tile_pipe_barrier();
        // buf_out = x * hardsigmoid(x)
        tile_std::__tile_v_mul_f32(buf_out, buf_in, buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
