// build-pass

// Broadcast/elementwise operation kernels.
// Maps to MultiKernelBench/reference/broadcast/ category:
//   add_bias, elementwise_mul, division, subtract, max, clamp

#![feature(no_core)]

#![no_std]
#![no_core]

/// add_bias_broadcast: y = x + bias (scalar)
/// Maps to broadcast/add_bias_broadcast.py
#[tile_std::tile_kernel]
pub fn add_bias(input: *const f32, output: *mut f32, bias_buf: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bias = *bias_buf;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_adds_f32(buf_out, buf_in, bias, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// elementwise_mul_broadcast: z = x * y
/// Maps to broadcast/elmentwise_mul_broadcast.py
#[tile_std::tile_kernel]
pub fn elementwise_mul(x: *const f32, y: *const f32, z: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_mul_f32(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(z, bz, n);
    }
}

/// division_broadcast: z = x / y
/// Maps to broadcast/division_broadcast.py
#[tile_std::tile_kernel]
pub fn elementwise_div(x: *const f32, y: *const f32, z: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_div_f32(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(z, bz, n);
    }
}

/// subtract_with_bias_broadcast: z = x - y
/// Maps to broadcast/subtract_with_bias_broadcast.py
#[tile_std::tile_kernel]
pub fn elementwise_sub(x: *const f32, y: *const f32, z: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_sub_f32(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(z, bz, n);
    }
}

/// max_broadcast: z = max(x, y)
/// Maps to broadcast/max_broadcast.py
#[tile_std::tile_kernel]
pub fn elementwise_max(x: *const f32, y: *const f32, z: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_max_f32(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(z, bz, n);
    }
}

/// clamp_broadcast: y = clamp(x, min_val, max_val)
/// Maps to broadcast/clamp_broadcast.py
#[tile_std::tile_kernel]
pub fn clamp(input: *const f32, output: *mut f32, bounds: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let min_val = *bounds;
        let max_val = *bounds.wrapping_add(1);
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::hardtanh_f32(buf_out, buf_in, min_val, max_val, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

/// elementwise_min: z = min(x, y)
#[tile_std::tile_kernel]
pub fn elementwise_min(x: *const f32, y: *const f32, z: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(by, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_min_f32(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(z, bz, n);
    }
}

/// power_broadcast: y = x^2 (element-wise square)
/// Maps to broadcast/power_broadcast.py (simplified to square)
#[tile_std::tile_kernel]
pub fn elementwise_square(input: *const f32, output: *mut f32, len: *const u32) {
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
