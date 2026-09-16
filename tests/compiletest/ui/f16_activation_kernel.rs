// build-pass

// Half-precision (f16) activation kernels.
// Many MultiKernelBench kernels operate on f16 data.

#![feature(no_core)]

#![no_std]
#![no_core]

/// f16 ReLU: relu(x) = max(x, 0)
#[tile_std::tile_kernel]
pub fn relu_f16(input: *const u16, output: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_maxs_f16(buf_out, buf_in, 0.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(output, buf_out, n);
    }
}

/// f16 sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
#[tile_std::tile_kernel]
pub fn sigmoid_f16(input: *const u16, output: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // dst = -x
        tile_std::__tile_muls_f16(buf_out, buf_in, -1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // dst = exp(-x)
        tile_std::__tile_v_exp_f16(buf_out, buf_out, n);
        tile_std::__tile_pipe_barrier();
        // dst = 1 + exp(-x)
        tile_std::__tile_adds_f16(buf_out, buf_out, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // dst = 1/(1+exp(-x))
        tile_std::__tile_reciprocal_f16(buf_out, buf_out, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(output, buf_out, n);
    }
}

/// f16 abs: abs(x) = |x|
#[tile_std::tile_kernel]
pub fn abs_f16(input: *const u16, output: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_abs_f16(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(output, buf_out, n);
    }
}

/// f16 exp: exp(x) = e^x
#[tile_std::tile_kernel]
pub fn exp_f16(input: *const u16, output: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_exp_f16(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(output, buf_out, n);
    }
}

/// f16 ln: ln(x) = log(x)
#[tile_std::tile_kernel]
pub fn ln_f16(input: *const u16, output: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_ln_f16(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(output, buf_out, n);
    }
}

/// f16 sqrt: sqrt(x)
#[tile_std::tile_kernel]
pub fn sqrt_f16(input: *const u16, output: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_sqrt_f16(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(output, buf_out, n);
    }
}

/// f16 rsqrt: rsqrt(x) = 1/sqrt(x)
#[tile_std::tile_kernel]
pub fn rsqrt_f16(input: *const u16, output: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_rsqrt_f16(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(output, buf_out, n);
    }
}

/// f16 reciprocal: reciprocal(x) = 1/x
#[tile_std::tile_kernel]
pub fn reciprocal_f16(input: *const u16, output: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_reciprocal_f16(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(output, buf_out, n);
    }
}

/// f16 vec_add: z = x + y
#[tile_std::tile_kernel]
pub fn vec_add_f16(x: *const u16, y: *const u16, z: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(bx, x, n);
        tile_std::__tile_buf_load_f16(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_add_f16(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(z, bz, n);
    }
}

/// f16 vec_sub: z = x - y
#[tile_std::tile_kernel]
pub fn vec_sub_f16(x: *const u16, y: *const u16, z: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(bx, x, n);
        tile_std::__tile_buf_load_f16(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_sub_f16(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(z, bz, n);
    }
}

/// f16 vec_mul: z = x * y
#[tile_std::tile_kernel]
pub fn vec_mul_f16(x: *const u16, y: *const u16, z: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(bx, x, n);
        tile_std::__tile_buf_load_f16(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_mul_f16(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(z, bz, n);
    }
}

/// f16 vec_div: z = x / y
#[tile_std::tile_kernel]
pub fn vec_div_f16(x: *const u16, y: *const u16, z: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(bx, x, n);
        tile_std::__tile_buf_load_f16(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_div_f16(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f16(z, bz, n);
    }
}

/// f16 reduce_max
#[tile_std::tile_kernel]
pub fn reduce_max_f16(input: *const u16, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::__tile_reduce_max_f16(buf_work, buf_in, buf_tmp, n);

        *output = result;
    }
}

/// f16 reduce_sum: load f16, cast to f32, ReduceSum in f32 precision
/// (ReduceSum on f16 buffers outputs zero on 910B — hardware limitation)
#[tile_std::tile_kernel]
pub fn reduce_sum_f16(input: *const u16, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_f32 = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f16(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // Cast f16 → f32, then reduce in f32 precision
        tile_std::__tile_cast_f16_to_f32(buf_f32, buf_in, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::__tile_v_reduce_sum_f32(buf_work, buf_f32, buf_tmp, n);

        *output = result;
    }
}
