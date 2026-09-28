// Case 5: Double-Free — Prevented by No Free Operation in API
//
// In tile-rs, there is no FreeTensor() or equivalent. Buffer IDs
// are plain u32 integers that name a buffer slot — they are not
// owning handles that can be "freed". The code generator manages
// buffer lifetimes automatically.
//
// "Double-freeing" a buffer ID is meaningless: it's like "freeing"
// the number 3 twice. Buffer IDs have no ownership semantics at the
// API level, so the entire class of double-free bugs is eliminated.

#![feature(no_core)]
#![no_std]
#![no_core]

/// Vector addition kernel — no free operations needed.
///
/// Buffer IDs (buf_x, buf_y, buf_z) are allocated once and reused
/// across all tile iterations. No manual lifecycle management means
/// no double-free, no use-after-free, no leak.
#[tile_std::tile_kernel]
pub fn vec_add(x: *const u16, y: *const u16, z: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;
        let base = block_idx * n;

        let tile_size = 256u32;

        // Allocate buffers once. These IDs are valid for the entire kernel.
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

            tile_std::__tile_buf_load_f16(buf_x, x.wrapping_add(gm_off), len);
            tile_std::__tile_buf_load_f16(buf_y, y.wrapping_add(gm_off), len);
            tile_std::__tile_pipe_barrier();

            tile_std::__tile_v_add_f16(buf_z, buf_x, buf_y, len);
            tile_std::__tile_pipe_barrier();

            tile_std::__tile_buf_store_f16(z.wrapping_add(gm_off), buf_z, len);

            // No FreeTensor here. buf_x and buf_y are reused on the next
            // iteration. Even if this line were duplicated by copy-paste,
            // there is simply no free function to call.

            offset = offset + tile_size;
        }

        // Kernel returns — all buffers implicitly released.
    }
}
