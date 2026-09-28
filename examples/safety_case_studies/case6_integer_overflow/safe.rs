// Case 6: Integer Overflow — Made Explicit by wrapping_mul
//
// In Rust, unsigned integer overflow in release builds wraps silently
// (same as C++), but the language provides wrapping_mul/wrapping_add
// to make the intent explicit. In debug builds, regular arithmetic
// (`*`, `+`) panics on overflow, catching the bug during testing.
//
// The key difference is intentionality:
//   - C++: `blockIdx * perBlockLen` silently wraps, developer unaware
//   - Rust: `wrapping_mul` documents that overflow is expected/handled
//   - Rust debug: `blockIdx * perBlockLen` panics, catching the bug early
//
// For NPU kernels (compiled in release mode), wrapping_mul serves as
// documentation that the developer has considered overflow. Regular `*`
// would also wrap in release, but the convention of using wrapping_mul
// flags the operation for review.

#![feature(no_core)]
#![no_std]
#![no_core]

/// Vector addition with explicit overflow handling in offset calculation.
///
/// wrapping_mul documents that this multiplication may overflow for
/// large tensor sizes. A reviewer seeing wrapping_mul knows to check
/// whether the overflow is actually safe for the use case.
///
/// In contrast, the C++ version uses plain `*` which silently wraps
/// with no indication that overflow was considered.
#[tile_std::tile_kernel]
pub fn vec_add(x: *const u16, y: *const u16, z: *mut u16, len: *const u32) {
    unsafe {
        let n = *len;
        let block_idx = tile_std::get_block_idx() as u32;

        // wrapping_mul makes overflow semantics explicit.
        // A developer reading this line knows that:
        //   1. This multiplication CAN overflow for large inputs
        //   2. The overflow behavior is intentionally wrapping
        //   3. This is a potential correctness concern worth reviewing
        //
        // In debug builds (used for CPU-side testing), plain `*` would
        // panic on overflow, catching the issue during development:
        //   let offset = block_idx * n;  // panics in debug if overflows!
        let offset = block_idx.wrapping_mul(n);

        let tile_size = 256u32;
        let buf_x = tile_std::__tile_buf_alloc(tile_size);
        let buf_y = tile_std::__tile_buf_alloc(tile_size);
        let buf_z = tile_std::__tile_buf_alloc(tile_size);

        let mut tile_off = 0u32;
        loop {
            if tile_off >= n {
                break;
            }
            let mut len = tile_size;
            if tile_off + len > n {
                len = n - tile_off;
            }
            let gm_off = (offset.wrapping_add(tile_off)) as usize;

            tile_std::__tile_buf_load_f16(buf_x, x.wrapping_add(gm_off), len);
            tile_std::__tile_buf_load_f16(buf_y, y.wrapping_add(gm_off), len);
            tile_std::__tile_pipe_barrier();

            tile_std::__tile_v_add_f16(buf_z, buf_x, buf_y, len);
            tile_std::__tile_pipe_barrier();

            tile_std::__tile_buf_store_f16(z.wrapping_add(gm_off), buf_z, len);

            tile_off = tile_off + tile_size;
        }
    }
}
