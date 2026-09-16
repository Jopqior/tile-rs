// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Basic array repeat initialization
fn test_repeat_i32() {
    let arr = [42i32; 4];
    assert_eq!(arr[0], 42);
    assert_eq!(arr[1], 42);
    assert_eq!(arr[2], 42);
    assert_eq!(arr[3], 42);
}

fn test_repeat_zero() {
    let arr = [0u32; 8];
    let mut sum = 0u32;
    let mut i = 0usize;
    loop {
        if i >= 8 { break; }
        sum = sum + arr[i];
        i = i + 1;
    }
    assert_eq!(sum, 0);
}

fn test_repeat_u8() {
    let arr = [0xFFu8; 3];
    assert_eq!(arr[0], 255);
    assert_eq!(arr[1], 255);
    assert_eq!(arr[2], 255);
}

fn test_repeat_bool() {
    let arr = [true; 4];
    assert!(arr[0]);
    assert!(arr[1]);
    assert!(arr[2]);
    assert!(arr[3]);
}

fn test_repeat_f32() {
    let arr = [1.5f32; 3];
    let mut sum = 0.0f32;
    let mut i = 0usize;
    loop {
        if i >= 3 { break; }
        sum = sum + arr[i];
        i = i + 1;
    }
    // 1.5 * 3 = 4.5
    assert!(sum > 4.4 && sum < 4.6);
}

// Repeat with computed value
fn test_repeat_computed() {
    let val = 3i32 * 7;
    let arr = [val; 5];
    let mut i = 0usize;
    loop {
        if i >= 5 { break; }
        assert_eq!(arr[i], 21);
        i = i + 1;
    }
}

// Single element repeat
fn test_repeat_single() {
    let arr = [99i32; 1];
    assert_eq!(arr[0], 99);
}

// Array of pairs (struct-like)
fn test_repeat_negative() {
    let arr = [-1i32; 4];
    let mut sum = 0i32;
    let mut i = 0usize;
    loop {
        if i >= 4 { break; }
        sum = sum + arr[i];
        i = i + 1;
    }
    assert_eq!(sum, -4);
}
