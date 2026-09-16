// build-pass

// Softplus activation kernel: softplus(x) = ln(1 + exp(x))
// Composed from: Exp -> Adds(1) -> Ln

#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn softplus(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // buf_out = exp(x)
        tile_std::__tile_v_exp_f32(buf_out, buf_in, n);
        tile_std::__tile_pipe_barrier();
        // buf_out = 1 + exp(x)
        tile_std::__tile_adds_f32(buf_out, buf_out, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // buf_out = ln(1 + exp(x))
        tile_std::__tile_ln_f32(buf_out, buf_out, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
