// build-pass

// Reduction operation kernels.
// Maps to MultiKernelBench/reference/reduce/ category.
// Output is broadcast to a UB buffer and DMA-stored (scalar GM writes don't work on NPU).

#![feature(no_core)]

#![no_std]
#![no_core]

/// Max reduction: y = max(x)
/// Maps to reduce/max_reduction_over_a_dimension.py
#[tile_std::tile_kernel]
pub fn reduce_max(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::__tile_v_reduce_max_f32(buf_work, buf_in, buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_work, buf_work, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_work, buf_work, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_work, n);
    }
}

/// Min reduction: y = min(x)
/// Maps to reduce/min_reduction_over_a_dimension.py
#[tile_std::tile_kernel]
pub fn reduce_min(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::__tile_reduce_min_f32(buf_work, buf_in, buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_work, buf_work, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_work, buf_work, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_work, n);
    }
}

/// Sum reduction: y = sum(x)
/// Maps to reduce/sum_reduction_over_a_dimension.py
#[tile_std::tile_kernel]
pub fn reduce_sum(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::__tile_v_reduce_sum_f32(buf_work, buf_in, buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_work, buf_work, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_work, buf_work, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_work, n);
    }
}

/// Mean reduction: y = mean(x) = sum(x) / n
/// Maps to reduce/mean_reduction_over_a_dimension.py
/// Uses scalar division (sum / n) which works on 310P (confirmed by mse_loss).
#[tile_std::tile_kernel]
pub fn reduce_mean(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);
        let buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let sum = tile_std::__tile_v_reduce_sum_f32(buf_work, buf_in, buf_tmp, n);

        // mean = sum / n (scalar division — works on 310P)
        let mean = sum / (n as f32);

        // Broadcast mean to buf_work for DMA store
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_work, buf_work, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_work, buf_work, mean, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_work, n);
    }
}

/// Product reduction: y = prod(x)
/// Maps to reduce/product_reduction_over_a_dimension.py
/// Computed as exp(sum(log(x))) — only correct for positive inputs.
#[tile_std::tile_kernel]
pub fn reduce_prod(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::kernel_ops::reduce_prod_f32(&mut buf_work, &mut buf_in, &mut buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf_work, buf_work, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_work, buf_work, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_work, n);
    }
}
