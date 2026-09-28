// build-pass

// Pooling-related operations (1D element-wise forms).
// Maps to MultiKernelBench/reference/pooling/ category.
// Full 2D pooling requires index ops; these implement the reduction parts.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Global average pooling (= reduce mean)
/// Maps to pooling/avg_pool.py (global case)
#[tile_std::tile_kernel]
pub fn global_avg_pool(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);
        *output = mean;
    }
}

/// Global max pooling (= reduce max)
/// Maps to pooling/max_pool.py (global case)
#[tile_std::tile_kernel]
pub fn global_max_pool(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        let max_val = tile_std::__tile_v_reduce_max_f32(work, buf, tmp, n);
        *output = max_val;
    }
}

/// Global min pooling (= reduce min)
#[tile_std::tile_kernel]
pub fn global_min_pool(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        let min_val = tile_std::__tile_reduce_min_f32(work, buf, tmp, n);
        *output = min_val;
    }
}

/// Avg pool + sigmoid (post-pooling activation)
/// Maps to fuse/conv2d_avg_pool_sigmoid_sum.py (partial)
#[tile_std::tile_kernel]
pub fn fused_avgpool_sigmoid(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // "avg pool" = mean over entire vector
        let mean = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);

        // Apply sigmoid to mean
        let neg_mean = -mean;
        let sig = 1.0f32 / (1.0f32 + tile_std::core::builtins::expf(neg_mean));
        *output = sig;
    }
}

/// Avg pool + sigmoid + sum
/// Maps to fuse/conv2d_avg_pool_sigmoid_sum.py
#[tile_std::tile_kernel]
pub fn fused_pool_sigmoid_sum(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let work = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // sigmoid
        tile_std::kernel_ops::sigmoid_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // sum
        let sum = tile_std::__tile_v_reduce_sum_f32(buf, buf, tmp, n);
        *output = sum;
    }
}

/// LP pooling (p=2): output = sqrt(mean(x^2))
/// This is equivalent to RMS (root mean square)
#[tile_std::tile_kernel]
pub fn lp_pool_2(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();

        // x^2
        tile_std::__tile_v_mul_f32(buf, buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // mean(x^2)
        let mean_sq = tile_std::kernel_ops::reduce_mean_f32(&mut work, &buf, &mut tmp, n);
        // sqrt(mean(x^2))
        *output = tile_std::core::builtins::sqrtf(mean_sq);
    }
}
