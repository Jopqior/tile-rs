// build-pass

// Scalar multiply kernel: y = alpha * x
// Maps directly to AscendC::Muls (scalar-vector multiply)

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn scalar_mul(
    input: *const f32,
    output: *mut f32,
    scalar: *const f32,
    len: *const u32,
) {
    unsafe {
        let n = *len;
        let alpha = *scalar;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(buf_out, buf_in, alpha, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
