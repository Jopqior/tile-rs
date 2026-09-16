// build-pass

// f32 unary vector operation kernels.
// Covers fundamental operations used across all categories.

#![feature(no_core)]

#![no_std]
#![no_core]

/// exp: y = e^x
#[tile_std::tile_kernel]
pub fn exp_f32(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_exp_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// log: y = ln(x)
#[tile_std::tile_kernel]
pub fn ln_f32(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_ln_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// sqrt: y = sqrt(x)
#[tile_std::tile_kernel]
pub fn sqrt_f32(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_sqrt_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// rsqrt: y = 1/sqrt(x)
#[tile_std::tile_kernel]
pub fn rsqrt_f32(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_rsqrt_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// reciprocal: y = 1/x
#[tile_std::tile_kernel]
pub fn reciprocal_f32(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_reciprocal_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// negate: y = -x
#[tile_std::tile_kernel]
pub fn negate_f32(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(buf_out, buf_in, -1.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// square: y = x^2
#[tile_std::tile_kernel]
pub fn square_f32(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_mul_f32(buf_out, buf_in, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// cube: y = x^3
#[tile_std::tile_kernel]
pub fn cube_f32(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // x^2 — squaring (all same input), safe
        tile_std::__tile_v_mul_f32(buf_out, buf_in, buf_in, n);
        tile_std::__tile_pipe_barrier();
        // x^3 = x^2 * x — all separate (buf_tmp != buf_out != buf_in)
        tile_std::__tile_v_mul_f32(buf_tmp, buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_tmp, n);
    }
}
