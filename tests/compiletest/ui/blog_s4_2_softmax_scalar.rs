// build-pass

// Blog section 4.2: Scalar softmax kernel
// Exact copy from blog/ascend-rs-memory-safe-npu-programming-en.md section 4.2

#![feature(no_core)]
#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn softmax(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len as usize;

        // Step 1: Find max value for numerical stability
        let mut max_val = *input;
        let mut i = 1usize;
        loop {
            if i >= n { break; }
            let val = *input.wrapping_add(i);
            if val > max_val { max_val = val; }
            i = i + 1;
        }

        // Step 2: Compute exp(x_i - max) and accumulate sum
        let mut sum: f32 = 0.0;
        i = 0;
        loop {
            if i >= n { break; }
            let exp_val = (*input.wrapping_add(i) - max_val).exp();
            *output.wrapping_add(i) = exp_val;
            sum = sum + exp_val;
            i = i + 1;
        }

        // Step 3: Normalize
        i = 0;
        loop {
            if i >= n { break; }
            *output.wrapping_add(i) = *output.wrapping_add(i) / sum;
            i = i + 1;
        }
    }
}
