//! CPU reference implementations of extended math kernels.
//! Mirrors: math_ext_kernel.rs

/// Masked cumulative sum: cumsum of x where mask is true.
pub fn masked_cumsum(input: &[f32], mask: &[bool], output: &mut [f32]) {
    let mut acc = 0.0f32;
    for i in 0..input.len() {
        if mask[i] {
            acc += input[i];
        }
        output[i] = acc;
    }
}
