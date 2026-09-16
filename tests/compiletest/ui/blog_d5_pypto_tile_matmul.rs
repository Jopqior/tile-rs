// build-pass

// Blog Appendix D.5: PyPTO tile ops → tile-rs matmul kernel
// From blog/ascend-rs-memory-safe-npu-programming-en.md Appendix D.5

#![feature(no_core)]
#![no_std]
#![no_core]

// Rust kernel: PyPTO tile ops → tile-rs instead of raw C++
#[tile_std::tile_kernel]
pub fn pypto_tile_matmul(
    a: *const u16, b: *const u16, c: *mut f32, n_ptr: *const u32,
) {
    unsafe {
        let n = *n_ptr;
        // Typed buffer allocation — codegen maps to TBuf with correct TPosition
        let l1_a  = tile_std::__tile_buf_alloc_l1(n);   // L1 buffer
        let l0a   = tile_std::__tile_buf_alloc_l0a(n);  // L0A buffer (cube input A)
        let l0b   = tile_std::__tile_buf_alloc_l0b(n);  // L0B buffer (cube input B)
        let l0c   = tile_std::__tile_buf_alloc_l0c(n);  // L0C buffer (cube output)

        // Each alloc maps to a specific TBuf<TPosition::*> in codegen
        // L0A → TBuf<TPosition::A1>, L0B → TBuf<TPosition::B1>, etc.
        // Mixing positions is a compile error in the generated C++
        tile_std::__tile_mmad_f16(l0c, l0a, l0b, n, n, n, 1);
    }
}
