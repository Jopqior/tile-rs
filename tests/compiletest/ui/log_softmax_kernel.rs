// build-pass

// LogSoftmax kernel: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
// Maps to MultiKernelBench/reference/activation/log_softmax.py

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn log_softmax(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);
        let mut buf_work2 = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::log_softmax_f32(&mut buf_out, &mut buf_in, &mut buf_work, &mut buf_work2, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
