// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Byte array constants (string-like data)
const HELLO: [u8; 5] = [72, 101, 108, 108, 111]; // "Hello"
const DIGITS: [u8; 10] = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]; // "0123456789"

// Static byte arrays
static LOOKUP: [u8; 4] = [0, 1, 2, 3];
static FLAGS: [u8; 8] = [1, 0, 1, 0, 1, 0, 1, 0];

fn test_const_byte_array() {
    assert_eq!(HELLO[0], 72);  // 'H'
    assert_eq!(HELLO[4], 111); // 'o'
}

fn test_const_digit_lookup() {
    assert_eq!(DIGITS[0], 48); // '0'
    assert_eq!(DIGITS[9], 57); // '9'
}

fn test_static_byte_lookup() {
    assert_eq!(LOOKUP[0], 0);
    assert_eq!(LOOKUP[3], 3);
}

fn test_static_flags() {
    let mut count = 0u8;
    let mut i = 0usize;
    loop {
        if i >= 8 { break; }
        count = count + FLAGS[i];
        i = i + 1;
    }
    assert_eq!(count, 4);
}

fn test_byte_array_sum() {
    let mut sum = 0u32;
    let mut i = 0usize;
    loop {
        if i >= 5 { break; }
        sum = sum + HELLO[i] as u32;
        i = i + 1;
    }
    // 72 + 101 + 108 + 108 + 111 = 500
    assert_eq!(sum, 500);
}
