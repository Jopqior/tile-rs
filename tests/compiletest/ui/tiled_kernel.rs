// build-pass

// Tiled kernel variants that process data in chunks.
// Demonstrates the tiling pattern critical for large inputs.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Tiled ReLU: processes input in 256-element tiles
#[tile_std::tile_kernel]
pub fn relu_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let buf = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_maxs_f32(buf, buf, 0.0f32, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled sigmoid
#[tile_std::tile_kernel]
pub fn sigmoid_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let buf = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::sigmoid_f32(buf, buf, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled GELU
#[tile_std::tile_kernel]
pub fn gelu_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let mut buf = tile_std::__tile_buf_alloc(tile_size);
        let mut buf_out = tile_std::__tile_buf_alloc(tile_size);
        let mut tmp = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::gelu_f32(&mut buf_out, &buf, &mut tmp, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf_out, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled tanh
#[tile_std::tile_kernel]
pub fn tanh_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let buf = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::tanh_f32(buf, buf, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled swish
#[tile_std::tile_kernel]
pub fn swish_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let mut buf = tile_std::__tile_buf_alloc(tile_size);
        let mut buf_out = tile_std::__tile_buf_alloc(tile_size);
        let mut tmp = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::swish_f32(&mut buf_out, &buf, &mut tmp, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf_out, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled exp
#[tile_std::tile_kernel]
pub fn exp_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let buf = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_v_exp_f32(buf, buf, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled vec_add f32
#[tile_std::tile_kernel]
pub fn vec_add_tiled(x: *const f32, y: *const f32, z: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let bx = tile_std::__tile_buf_alloc(tile_size);
        let by = tile_std::__tile_buf_alloc(tile_size);
        let bz = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(bx, x.wrapping_add(offset as usize), len);
            tile_std::__tile_buf_load_f32(by, y.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_v_add_f32(bz, bx, by, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(z.wrapping_add(offset as usize), bz, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled vec_mul f32
#[tile_std::tile_kernel]
pub fn vec_mul_tiled(x: *const f32, y: *const f32, z: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let bx = tile_std::__tile_buf_alloc(tile_size);
        let by = tile_std::__tile_buf_alloc(tile_size);
        let bz = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(bx, x.wrapping_add(offset as usize), len);
            tile_std::__tile_buf_load_f32(by, y.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_v_mul_f32(bz, bx, by, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(z.wrapping_add(offset as usize), bz, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled ELU
#[tile_std::tile_kernel]
pub fn elu_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let mut buf = tile_std::__tile_buf_alloc(tile_size);
        let mut tmp = tile_std::__tile_buf_alloc(tile_size);
        let mut work = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::elu_f32(&mut work, &mut buf, &mut tmp, 1.0f32, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), work, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled mish
#[tile_std::tile_kernel]
pub fn mish_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let mut buf = tile_std::__tile_buf_alloc(tile_size);
        let mut buf_out = tile_std::__tile_buf_alloc(tile_size);
        let mut tmp = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::mish_f32(&mut buf_out, &buf, &mut tmp, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf_out, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled layernorm
#[tile_std::tile_kernel]
pub fn layernorm_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let buf = tile_std::__tile_buf_alloc(tile_size);
        let mut buf_out = tile_std::__tile_buf_alloc(tile_size);
        let mut work = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::layernorm_f32(&mut buf_out, &buf, &mut work, len, 1e-5f32);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf_out, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled softmax (per-tile normalization)
#[tile_std::tile_kernel]
pub fn softmax_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let mut buf = tile_std::__tile_buf_alloc(tile_size);
        let mut buf_out = tile_std::__tile_buf_alloc(tile_size);
        let mut work = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::softmax_f32(&mut buf_out, &mut buf, &mut work, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf_out, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled SELU
#[tile_std::tile_kernel]
pub fn selu_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let mut buf = tile_std::__tile_buf_alloc(tile_size);
        let mut tmp = tile_std::__tile_buf_alloc(tile_size);
        let mut work = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::selu_f32(&mut work, &mut buf, &mut tmp, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), work, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled leaky_relu
#[tile_std::tile_kernel]
pub fn leaky_relu_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let mut buf = tile_std::__tile_buf_alloc(tile_size);
        let mut tmp = tile_std::__tile_buf_alloc(tile_size);
        let mut work = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::leaky_relu_f32(&mut work, &mut buf, &mut tmp, 0.01f32, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), work, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled hardswish
#[tile_std::tile_kernel]
pub fn hardswish_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let mut buf = tile_std::__tile_buf_alloc(tile_size);
        let mut buf_out = tile_std::__tile_buf_alloc(tile_size);
        let mut tmp = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::hardswish_f32(&mut buf_out, &buf, &mut tmp, len);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf_out, len);
            offset = offset + tile_size;
        }
    }
}

/// Tiled rms_norm
#[tile_std::tile_kernel]
pub fn rmsnorm_tiled(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let tile_size = 256u32;
        let buf = tile_std::__tile_buf_alloc(tile_size);
        let mut buf_out = tile_std::__tile_buf_alloc(tile_size);
        let mut work = tile_std::__tile_buf_alloc(tile_size);
        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            tile_std::__tile_buf_load_f32(buf, input.wrapping_add(offset as usize), len);
            tile_std::__tile_pipe_barrier();
            tile_std::kernel_ops::rms_norm_f32(&mut buf_out, &buf, &mut work, len, 1e-5f32);
            tile_std::__tile_pipe_barrier();
            tile_std::__tile_buf_store_f32(output.wrapping_add(offset as usize), buf_out, len);
            offset = offset + tile_size;
        }
    }
}
