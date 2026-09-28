// build-pass

// Extended loss function kernels.
// Maps to MultiKernelBench/reference/loss/ category (remaining ops).

#![feature(no_core)]

#![no_std]
#![no_core]

/// Triplet margin loss: loss = max(0, ||a - p||^2 - ||a - n||^2 + margin)
/// Maps to loss/triplet_margin_loss.py
#[tile_std::tile_kernel]
pub fn triplet_margin_loss(
    anchor: *const f32, positive: *const f32, negative: *const f32,
    output: *mut f32, params: *const f32,
) {
    unsafe {
        let margin = *params;
        let n_ptr = params.wrapping_add(1) as *const u32;
        let n = *n_ptr;

        // Compute ||a - p||^2
        let mut dist_ap = 0.0f32;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let d = *anchor.wrapping_add(i as usize) - *positive.wrapping_add(i as usize);
            dist_ap = dist_ap + d * d;
            i = i + 1;
        }

        // Compute ||a - n||^2
        let mut dist_an = 0.0f32;
        i = 0;
        loop {
            if i >= n { break; }
            let d = *anchor.wrapping_add(i as usize) - *negative.wrapping_add(i as usize);
            dist_an = dist_an + d * d;
            i = i + 1;
        }

        // loss = max(0, dist_ap - dist_an + margin)
        let loss = dist_ap - dist_an + margin;
        if loss > 0.0f32 {
            *output = loss;
        } else {
            *output = 0.0f32;
        }
    }
}
