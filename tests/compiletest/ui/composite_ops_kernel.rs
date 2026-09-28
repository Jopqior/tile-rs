// build-pass

// Tests composite operations from tile_std::kernel_ops.
// Each kernel uses a high-level helper that internally chains
// vector intrinsics with proper pipe_barrier synchronization.

#![feature(no_core)]

#![no_std]
#![no_core]

// --- Sigmoid using composite helper ---
#[tile_std::tile_kernel]
pub fn test_sigmoid(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::sigmoid_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

// --- Tanh using composite helper ---
#[tile_std::tile_kernel]
pub fn test_tanh(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::tanh_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

// --- GELU using composite helper ---
#[tile_std::tile_kernel]
pub fn test_gelu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::gelu_f32(&mut buf_out, &buf_in, &mut buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}

// --- Softmax using composite helper ---
#[tile_std::tile_kernel]
pub fn test_softmax(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::softmax_f32(&mut buf_out, &mut buf_in, &mut buf_work, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
