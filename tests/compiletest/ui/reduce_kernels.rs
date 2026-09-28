// build-pass

// Reduction kernels: sum, max, min
// Uses AscendC ReduceSum/ReduceMax/ReduceMin intrinsics
// Output is broadcast to a UB buffer and DMA-stored (scalar GM writes don't work on NPU).

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn reduce_sum(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_dst = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let sum = tile_std::__tile_v_reduce_sum_f32(buf_dst, buf_in, buf_work, n);

        tile_std::__tile_pipe_barrier();
        // Broadcast scalar to buffer: zero then add
        tile_std::__tile_muls_f32(buf_dst, buf_dst, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_dst, buf_dst, sum, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_dst, n);
    }
}

#[tile_std::tile_kernel]
pub fn reduce_max(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_dst = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let max_val = tile_std::__tile_v_reduce_max_f32(buf_dst, buf_in, buf_work, n);

        tile_std::__tile_pipe_barrier();
        // Broadcast scalar to buffer: zero then add
        tile_std::__tile_muls_f32(buf_dst, buf_dst, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_dst, buf_dst, max_val, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_dst, n);
    }
}

#[tile_std::tile_kernel]
pub fn reduce_min(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_dst = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        let min_val = tile_std::__tile_reduce_min_f32(buf_dst, buf_in, buf_work, n);

        tile_std::__tile_pipe_barrier();
        // Broadcast scalar to buffer: zero then add
        tile_std::__tile_muls_f32(buf_dst, buf_dst, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf_dst, buf_dst, min_val, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_dst, n);
    }
}
