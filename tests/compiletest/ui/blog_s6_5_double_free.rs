// build-pass

// Blog section 6.5: Double-free case study (safe Rust version)
// Exact copy from blog/ascend-rs-memory-safe-npu-programming-en.md section 6.5

#![feature(no_core)]
#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn vec_add(x: *const u16, y: *const u16, z: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;
        let tile_size = 256u32;

        let buf_x = tile_std::__tile_buf_alloc(tile_size);
        let buf_y = tile_std::__tile_buf_alloc(tile_size);
        let buf_z = tile_std::__tile_buf_alloc(tile_size);

        let mut offset = 0u32;
        loop {
            if offset >= n { break; }
            let mut len = tile_size;
            if offset + len > n { len = n - offset; }
            let gm_off = (base + offset) as usize;

            tile_std::__tile_buf_load_f16(buf_x, x.wrapping_add(gm_off), len);
            tile_std::__tile_buf_load_f16(buf_y, y.wrapping_add(gm_off), len);
            tile_std::__tile_pipe_barrier();

            tile_std::__tile_v_add_f16(buf_z, buf_x, buf_y, len);
            tile_std::__tile_pipe_barrier();

            tile_std::__tile_buf_store_f16(z.wrapping_add(gm_off), buf_z, len);

            offset = offset + tile_size;
        }
    }
}
