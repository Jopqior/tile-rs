// build-pass

// Leaky ReLU activation kernel: leaky_relu(x) = max(x, 0) + alpha * min(x, 0)
// Uses two buffers to compute positive and negative parts separately.

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn leaky_relu(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let alpha = 0.01f32;

        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_pos = tile_std::__tile_buf_alloc(n);
        let mut buf_neg = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::leaky_relu_f32(&mut buf_pos, &mut buf_in, &mut buf_neg, alpha, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_pos, n);
    }
}
