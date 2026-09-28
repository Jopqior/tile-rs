// Case 1: Type Confusion — Prevented by Typed Function Signatures
//
// In tile-rs, the kernel's function signature encodes the element type
// of each tensor pointer. The host code must pass pointers of the correct
// type — passing *const u16 (f16) where *const f32 is expected is a
// compile-time error.
//
// This eliminates the entire class of GM_ADDR type confusion bugs at the
// language level: no runtime check needed, no cast to get wrong.

#![feature(no_core)]
#![no_std]
#![no_core]

/// Softmax kernel operating on f32 data.
///
/// The signature `input: *const f32` means the host MUST pass an f32 tensor.
/// If the host has f16 data (*const u16), calling this function is a type error:
///
///     softmax(f16_ptr, ...)  // ERROR: expected *const f32, found *const u16
///
/// Compare with AscendC where GM_ADDR erases all type information and the
/// kernel must cast manually — getting the cast wrong produces silent corruption.
#[tile_std::tile_kernel]
pub fn softmax(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;

        let buf_in = tile_std::__tile_buf_alloc(n);
        let buf_out = tile_std::__tile_buf_alloc(n);
        let buf_work = tile_std::__tile_buf_alloc(n);

        // Load f32 data — the _f32 suffix matches the pointer type.
        // There is no way to accidentally load f16 data through an f32 API.
        tile_std::__tile_buf_load_f32(buf_in, input, n);
        tile_std::__tile_pipe_barrier();

        // softmax_f32 expects f32 buffers — type consistency is maintained
        // throughout the entire pipeline without manual casts.
        tile_std::kernel_ops::softmax_f32(buf_out, buf_in, buf_work, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf_out, n);
    }
}
