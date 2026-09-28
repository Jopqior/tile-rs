// build-pass

// Extended math operation kernels.
// Maps to MultiKernelBench/reference/math/ category (remaining ops).

#![feature(no_core)]

#![no_std]
#![no_core]

/// Masked cumulative sum: cumsum(x * mask)
/// Maps to math/masked_cumsum.py
#[tile_std::tile_kernel]
pub fn masked_cumsum(
    input: *const f32, mask: *const u32, output: *mut f32, len: *const u32,
) {
    unsafe {
        let n = *len;
        let mut acc = 0.0f32;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let m = *mask.wrapping_add(i as usize);
            if m != 0 {
                acc = acc + *input.wrapping_add(i as usize);
            }
            *output.wrapping_add(i as usize) = acc;
            i = i + 1;
        }
    }
}
