// Case 2: Buffer Overflow — Prevented by Buffer-ID API
//
// In tile-rs, buffer operations work on opaque buffer IDs (u32), not
// raw pointers. Vector intrinsics take an explicit count parameter that
// specifies how many elements to process. There is no GetValue(i) or
// SetValue(i, v) with unchecked indexing.
//
// The count parameter naturally comes from the same variable used to
// allocate the buffer, making off-by-one errors structurally unlikely.
// Even if the count were wrong, the vector hardware operates on whole
// blocks rather than individual elements — there is no single-element
// indexing API to get wrong.

#![feature(no_core)]
#![no_std]
#![no_core]

/// Softmax kernel using vector operations with explicit count.
///
/// The count `n` passed to each vector op is the same value used to
/// allocate the buffer. There is no separate loop variable that could
/// drift out of sync. No element-wise indexing means no off-by-one.
#[tile_std::tile_kernel]
pub fn softmax(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // softmax_f32 operates on the entire buffer of `n` elements.
        // There is no loop index, no GetValue(i), no SetValue(i, v).
        // The count `n` is the same value used in __tile_buf_alloc —
        // the allocation and the operation are inherently consistent.
        tile_std::kernel_ops::softmax_f32(buf_out, buf_in, buf_work, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);

        // Compare with the C++ version where a scalar loop with
        // `for (i = 0; i <= len; i++)` writes one past the buffer.
        // That bug class simply does not exist in the buffer-ID API.
    }
}
