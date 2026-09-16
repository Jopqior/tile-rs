// build-pass

// Multi-block kernels that distribute work across AICore blocks.
// These demonstrate the block-level parallelism pattern used in
// production kernels.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Multi-block ReLU: each block processes a portion of the input
#[tile_std::tile_kernel]
pub fn relu_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_maxs_f32(buf_out, buf_in, 0.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block sigmoid
#[tile_std::tile_kernel]
pub fn sigmoid_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::sigmoid_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block GELU
#[tile_std::tile_kernel]
pub fn gelu_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::gelu_f32(&mut buf_out, &buf_in, &mut buf_tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block tanh
#[tile_std::tile_kernel]
pub fn tanh_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::tanh_f32(buf_out, buf_in, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block softmax
#[tile_std::tile_kernel]
pub fn softmax_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let mut buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::softmax_f32(&mut buf_out, &mut buf_in, &mut buf_work, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block layernorm
#[tile_std::tile_kernel]
pub fn layernorm_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf_in, &mut buf_work, n, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block vec_add (f32)
#[tile_std::tile_kernel]
pub fn vec_add_multiblock(x: *const f32, y: *const f32, z: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let bx = tile_std::__tile_buf_alloc(n);
        let by = tile_std::__tile_buf_alloc(n);
        let bz = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bx, x.wrapping_add(base as usize), n);
        tile_std::__tile_buf_load_f32(by, y.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_add_f32(bz, bx, by, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(z.wrapping_add(base as usize), bz, n);
    }
}

/// Multi-block mish
#[tile_std::tile_kernel]
pub fn mish_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::mish_f32(&mut buf_out, &buf, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block swish
#[tile_std::tile_kernel]
pub fn swish_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::swish_f32(&mut buf_out, &buf, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block ELU
#[tile_std::tile_kernel]
pub fn elu_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::elu_f32(&mut work, &mut buf, &mut tmp, 1.0f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), work, n);
    }
}

/// Multi-block SELU
#[tile_std::tile_kernel]
pub fn selu_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::selu_f32(&mut work, &mut buf, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), work, n);
    }
}

/// Multi-block leaky_relu
#[tile_std::tile_kernel]
pub fn leaky_relu_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::leaky_relu_f32(&mut work, &mut buf, &mut tmp, 0.01f32, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), work, n);
    }
}

/// Multi-block RMS norm
#[tile_std::tile_kernel]
pub fn rmsnorm_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::rms_norm_f32(&mut buf_out, &buf, &mut work, n, 1e-5f32);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block hardswish
#[tile_std::tile_kernel]
pub fn hardswish_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf = tile_std::__tile_buf_alloc(n);
        let mut buf_out = tile_std::__tile_buf_alloc(n);
        let mut tmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::hardswish_f32(&mut buf_out, &buf, &mut tmp, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf_out, n);
    }
}

/// Multi-block hardsigmoid
#[tile_std::tile_kernel]
pub fn hardsigmoid_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::hardsigmoid_f32(buf, buf, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf, n);
    }
}

/// Multi-block softplus
#[tile_std::tile_kernel]
pub fn softplus_multiblock(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let buf = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf, input.wrapping_add(base as usize), n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::softplus_f32(buf, buf, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output.wrapping_add(base as usize), buf, n);
    }
}
