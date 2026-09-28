// build-pass

// Math operation kernels.
// Maps to MultiKernelBench/reference/math/ category.
//
// Note: cumsum/cumprod kernels are in scalar_loop_kernels.rs (separate file)
// because they use GM pointer arithmetic in loops which generates gm_ptr_load
// placeholders that fail C++ compilation. Keeping them separate prevents
// matrix_scalar_mul from being blocked.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Matrix-scalar multiplication: C = A * s
/// Maps to math/matrix_scalar_multiplication.py
#[tile_std::tile_kernel]
pub fn matrix_scalar_mul(input: *const f32, output: *mut f32, scalar_buf: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let s = *scalar_buf;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(buf_out, buf_in, s, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
