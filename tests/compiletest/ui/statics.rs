// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Static constant values
static MAGIC: i32 = 42;
static PI_APPROX: f32 = 3.14159;

// Static const array (lookup table)
const TABLE: [i32; 5] = [10, 20, 30, 40, 50];

// Static with computed value
const COMPUTED: i32 = 6 * 7;

fn test_static_read() {
    let val = unsafe { core::ptr::read_volatile(&MAGIC as *const i32) };
    assert_eq!(val, 42);
}

fn test_const_table_access() {
    assert_eq!(TABLE[0], 10);
    assert_eq!(TABLE[2], 30);
    assert_eq!(TABLE[4], 50);
}

fn test_const_computed() {
    assert_eq!(COMPUTED, 42);
}

fn test_const_in_loop() {
    let mut sum = 0i32;
    let mut i = 0usize;
    loop {
        if i >= 5 { break; }
        sum = sum + TABLE[i];
        i = i + 1;
    }
    assert_eq!(sum, 150);
}

fn test_nested_const() {
    const INNER: [i32; 3] = [1, 2, 3];
    let mut product = 1i32;
    let mut i = 0usize;
    loop {
        if i >= 3 { break; }
        product = product * INNER[i];
        i = i + 1;
    }
    assert_eq!(product, 6);
}

// Static arrays (should get dense<[...]> initializer in MLIR)
static WEIGHTS: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
static OFFSETS: [i32; 3] = [100, 200, 300];
static BYTE_TABLE: [i8; 4] = [1, 2, 3, 4];

fn test_static_array_access() {
    assert_eq!(WEIGHTS[0], 0.25);
    assert_eq!(WEIGHTS[3], 1.0);
}

fn test_static_i32_array() {
    let mut sum = 0i32;
    let mut i = 0usize;
    loop {
        if i >= 3 { break; }
        sum = sum + OFFSETS[i];
        i = i + 1;
    }
    assert_eq!(sum, 600);
}

fn test_static_byte_array() {
    assert_eq!(BYTE_TABLE[0], 1);
    assert_eq!(BYTE_TABLE[3], 4);
}

fn test_static_array_as_lookup() {
    // Use static array as lookup table in a computation
    let idx = 2usize;
    let val = OFFSETS[idx];
    assert_eq!(val + 50, 350);
}
