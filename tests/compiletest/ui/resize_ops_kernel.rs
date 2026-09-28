// build-pass

// Resize/interpolation operations (element-wise approximations).
// Maps to MultiKernelBench/reference/resize/ category.
// Full 2D interpolation requires index ops not yet in tile_std;
// these implement the 1D/element-wise parts.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Nearest-neighbor resize (identity for element-wise: just copy with scaling)
/// Maps to resize/ category (base case)
#[tile_std::tile_kernel]
pub fn resize_nearest(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// Linear interpolation between two tensors: output = (1-t)*a + t*b
/// Maps to resize/ bilinear interpolation (1D case)
#[tile_std::tile_kernel]
pub fn lerp(a: *const f32, b: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let t = *config;
        let ba = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);
        let bout = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(ba, a, n);
        tile_std::__tile_buf_load_f32(bb, b, n);
        tile_std::__tile_pipe_barrier();

        // (1-t) * a
        tile_std::__tile_muls_f32(bout, ba, 1.0f32 - t, n);
        tile_std::__tile_pipe_barrier();
        // t * b
        tile_std::__tile_muls_f32(ba, bb, t, n);
        tile_std::__tile_pipe_barrier();
        // (1-t)*a + t*b — ba dead after
        tile_std::__tile_v_add_f32(ba, bout, ba, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, ba, n);
    }
}

/// Bicubic interpolation weight: w(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1 for |t|<=1
/// Simplified to compute the weight polynomial on a vector of distances.
#[tile_std::tile_kernel]
pub fn bicubic_weight(distances: *const f32, weights: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let t2 = tile_std::__tile_buf_alloc(n);
        let t3 = tile_std::__tile_buf_alloc(n);
        let out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, distances, n);
        tile_std::__tile_pipe_barrier();

        // |t|
        tile_std::__tile_v_abs_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // t^2
        tile_std::__tile_v_mul_f32(t2, buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // t^3
        tile_std::__tile_v_mul_f32(t3, t2, buf, n);
        tile_std::__tile_pipe_barrier();

        // w = (a+2)*t^3; a = -0.5 => (1.5)*t^3
        tile_std::__tile_muls_f32(out, t3, 1.5f32, n);
        tile_std::__tile_pipe_barrier();
        // w -= (a+3)*t^2 => w -= 2.5*t^2
        tile_std::__tile_muls_f32(t2, t2, 2.5f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_sub_f32(out, out, t2, n);
        tile_std::__tile_pipe_barrier();
        // w += 1
        tile_std::__tile_adds_f32(out, out, 1.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(weights, out, n);
    }
}

/// Weighted sum of two buffers (for interpolation):
///   output = w1*a + w2*b
#[tile_std::tile_kernel]
pub fn weighted_sum(a: *const f32, b: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let w1 = *config;
        let w2 = *config.wrapping_add(1);
        let ba = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(ba, a, n);
        tile_std::__tile_buf_load_f32(bb, b, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_muls_f32(ba, ba, w1, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bb, bb, w2, n);
        tile_std::__tile_pipe_barrier();
        // bb dead after add
        tile_std::__tile_v_add_f32(bb, ba, bb, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bb, n);
    }
}

/// Trilinear interpolation (1D case: weighted average of 2 endpoints)
#[tile_std::tile_kernel]
pub fn trilinear_1d(a: *const f32, b: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let alpha = *config;
        let ba = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(ba, a, n);
        tile_std::__tile_buf_load_f32(bb, b, n);
        tile_std::__tile_pipe_barrier();

        // (1-alpha)*a + alpha*b
        tile_std::__tile_muls_f32(ba, ba, 1.0f32 - alpha, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bb, bb, alpha, n);
        tile_std::__tile_pipe_barrier();
        // bb dead after add
        tile_std::__tile_v_add_f32(bb, ba, bb, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bb, n);
    }
}
