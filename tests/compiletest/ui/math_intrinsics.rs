// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

fn test_exp() {
    let x: f32 = 1.0;
    let y = x.exp();
    let _ = y;
}

fn test_ln() {
    let x: f32 = 2.718;
    let y = x.ln();
    let _ = y;
}

fn test_sqrt() {
    let x: f32 = 4.0;
    let y = x.sqrt();
    let _ = y;
}

fn test_softmax_pattern() {
    // Numerically stable softmax on a small array
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    // Step 1: find max
    let mut max_val = data[0];
    let mut i = 1usize;
    loop {
        if i >= 4 {
            break;
        }
        if data[i] > max_val {
            max_val = data[i];
        }
        i = i + 1;
    }

    // Step 2: compute exp(x - max) and sum
    let mut sum: f32 = 0.0;
    let mut exp_vals: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    i = 0;
    loop {
        if i >= 4 {
            break;
        }
        exp_vals[i] = (data[i] - max_val).exp();
        sum = sum + exp_vals[i];
        i = i + 1;
    }

    // Step 3: normalize
    i = 0;
    loop {
        if i >= 4 {
            break;
        }
        let _ = exp_vals[i] / sum;
        i = i + 1;
    }
}
