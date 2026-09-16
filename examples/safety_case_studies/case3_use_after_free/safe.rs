// Case 3: Use-After-Free — Prevented by No Manual Free in API
//
// In tile-rs, buffer IDs are plain u32 integers. There is no
// FreeTensor() operation in the API. The code generator's queue system
// manages buffer lifetimes automatically — buffers remain valid for the
// entire kernel invocation.
//
// Since buffer IDs are just numbers (not pointers), "using a buffer
// after it's freed" is not a meaningful concept. The ID 0 is always
// the same buffer slot regardless of when you access it.

#![feature(no_core)]
#![no_std]
#![no_core]

/// Vector addition kernel where buf_x remains valid throughout.
///
/// Compare with the C++ version where FreeTensor(xLocal) invalidates
/// the buffer, but xLocal.GetValue(0) still compiles and accesses
/// freed SRAM. Here, buf_x is just a u32 — it never becomes invalid.
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
            if offset >= n {
                break;
            }
            let mut len = tile_size;
            if offset + len > n {
                len = n - offset;
            }
            let gm_off = (base + offset) as usize;

            // Load input tiles
            tile_std::__tile_buf_load_f16(buf_x, x.wrapping_add(gm_off), len);
            tile_std::__tile_buf_load_f16(buf_y, y.wrapping_add(gm_off), len);
            tile_std::__tile_pipe_barrier();

            // Compute: buf_x is used here
            tile_std::__tile_v_add_f16(buf_z, buf_x, buf_y, len);
            tile_std::__tile_pipe_barrier();

            // No FreeTensor needed. buf_x, buf_y, buf_z are still valid.
            // The same buffer IDs are reused in the next tile iteration —
            // this is safe because the previous DMA store has completed
            // (guaranteed by pipe_barrier before the next load).

            tile_std::__tile_buf_store_f16(z.wrapping_add(gm_off), buf_z, len);

            offset = offset + tile_size;
        }

        // Kernel returns. All buffers are implicitly released.
        // No manual cleanup, no dangling references, no use-after-free.
    }
}
