// =============================================================================
// NPU Kernel: Softmax
// =============================================================================
//
// Numerically stable softmax: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// This kernel demonstrates math intrinsics (exp) on the Ascend NPU.
// Single-block execution for simplicity — all elements processed by one block.

#![feature(no_core)]
#![no_std]
#![no_core]

/// Diagnostic kernel: stores a constant to verify GM writes work.
#[tile_std::tile_kernel]
pub fn test_store_const(output: *mut f32) {
    unsafe {
        *output = 42.0f32;
    }
}

/// Diagnostic kernel: copies one f32 value from input to output.
#[tile_std::tile_kernel]
pub fn test_copy(input: *const f32, output: *mut f32) {
    unsafe {
        *output = *input;
    }
}

/// Softmax: output[i] = exp(input[i] - max(input)) / sum(exp(input[j] - max(input)))
///
/// Parameters:
///   - input: pointer to f32 input data on device
///   - output: pointer to f32 output data on device
///   - len: number of elements (passed as a single-element buffer)
#[tile_std::tile_kernel]
pub fn softmax(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len as usize;

        // Step 1: Find max value for numerical stability
        let mut max_val = *input;
        let mut i = 1usize;
        loop {
            if i >= n {
                break;
            }
            let val = *input.wrapping_add(i);
            if val > max_val {
                max_val = val;
            }
            i = i + 1;
        }

        // Step 2: Compute exp(x_i - max) and accumulate sum
        let mut sum: f32 = 0.0;
        i = 0;
        loop {
            if i >= n {
                break;
            }
            let exp_val = (*input.wrapping_add(i) - max_val).exp();
            *output.wrapping_add(i) = exp_val;
            sum = sum + exp_val;
            i = i + 1;
        }

        // Step 3: Normalize by dividing each element by sum
        i = 0;
        loop {
            if i >= n {
                break;
            }
            *output.wrapping_add(i) = *output.wrapping_add(i) / sum;
            i = i + 1;
        }
    }
}
