// build-pass

// Cumulative math operations (scalar loop GEP-DMA pattern).
// Maps to MultiKernelBench/reference/math/ category.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Cumulative product: output[i] = input[0] * input[1] * ... * input[i]
/// Maps to math/cumprod.py
#[tile_std::tile_kernel]
pub fn cumprod(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut acc = 1.0f32;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            acc = acc * *input.wrapping_add(i as usize);
            *output.wrapping_add(i as usize) = acc;
            i = i + 1;
        }
    }
}

/// Cumulative sum: output[i] = input[0] + input[1] + ... + input[i]
/// Maps to math/cumsum.py
#[tile_std::tile_kernel]
pub fn cumsum(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut acc = 0.0f32;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            acc = acc + *input.wrapping_add(i as usize);
            *output.wrapping_add(i as usize) = acc;
            i = i + 1;
        }
    }
}

/// Exclusive cumulative sum: output[i] = input[0] + ... + input[i-1], output[0] = 0
/// Maps to math/cumsum_exclusive.py
#[tile_std::tile_kernel]
pub fn cumsum_exclusive(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut acc = 0.0f32;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            *output.wrapping_add(i as usize) = acc;
            acc = acc + *input.wrapping_add(i as usize);
            i = i + 1;
        }
    }
}

/// Reverse cumulative sum: output[i] = input[i] + input[i+1] + ... + input[n-1]
/// Maps to math/cumsum_reverse.py
#[tile_std::tile_kernel]
pub fn cumsum_reverse(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut acc = 0.0f32;
        let mut i = n;
        loop {
            if i == 0 { break; }
            i = i - 1;
            acc = acc + *input.wrapping_add(i as usize);
            *output.wrapping_add(i as usize) = acc;
        }
    }
}
