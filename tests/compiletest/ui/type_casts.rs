// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Integer widening
fn test_widen_unsigned() {
    let a: u8 = 255;
    let b: u16 = a as u16;
    assert_eq!(b, 255u16);

    let c: u32 = b as u32;
    assert_eq!(c, 255u32);

    let d: u64 = c as u64;
    assert_eq!(d, 255u64);
}

fn test_widen_signed() {
    let a: i8 = -1;
    let b: i16 = a as i16;
    assert_eq!(b, -1i16);

    let c: i32 = b as i32;
    assert_eq!(c, -1i32);

    let d: i64 = c as i64;
    assert_eq!(d, -1i64);
}

// Integer narrowing (truncation)
fn test_narrow_unsigned() {
    let a: u32 = 256;
    let b: u8 = a as u8;
    assert_eq!(b, 0u8); // 256 truncated to 8 bits

    let c: u32 = 0xFF01;
    let d: u8 = c as u8;
    assert_eq!(d, 1u8);
}

fn test_narrow_signed() {
    let a: i32 = 128;
    let b: i8 = a as i8;
    assert_eq!(b, -128i8); // 128 wraps to -128 in i8

    let c: i32 = -1;
    let d: u8 = c as u8;
    assert_eq!(d, 255u8); // -1 as u8 = 255
}

// Signed/unsigned conversion
fn test_sign_cast() {
    let a: i32 = -1;
    let b: u32 = a as u32;
    assert_eq!(b, 4294967295u32); // 0xFFFFFFFF

    let c: u32 = 4294967295;
    let d: i32 = c as i32;
    assert_eq!(d, -1i32);
}

// Float to integer
fn test_float_to_int() {
    let a: f32 = 42.7;
    let b: i32 = a as i32;
    assert_eq!(b, 42);

    let c: f32 = -3.9;
    let d: i32 = c as i32;
    assert_eq!(d, -3);

    let e: f32 = 0.0;
    let f: i32 = e as i32;
    assert_eq!(f, 0);
}

// Integer to float
fn test_int_to_float() {
    let a: i32 = 42;
    let b: f32 = a as f32;
    assert_eq!(b, 42.0f32);

    let c: i32 = -10;
    let d: f32 = c as f32;
    assert_eq!(d, -10.0f32);
}

// Bool to integer
fn test_bool_to_int() {
    let t: bool = true;
    let f: bool = false;
    assert_eq!(t as i32, 1);
    assert_eq!(f as i32, 0);
    assert_eq!(t as u8, 1);
    assert_eq!(f as u8, 0);
}

// Chained casts
fn test_chained_casts() {
    let a: i8 = -1;
    let b: u64 = a as u64; // sign-extends to i64, then reinterprets as u64
    assert_eq!(b, 18446744073709551615u64); // 0xFFFFFFFFFFFFFFFF

    let c: u8 = 200;
    let d: i64 = c as i64; // zero-extends
    assert_eq!(d, 200i64);
}

// Cast in expressions
fn test_cast_in_expr() {
    let a: u8 = 100;
    let b: u8 = 200;
    // Avoid overflow by casting before add
    let sum: u32 = (a as u32) + (b as u32);
    assert_eq!(sum, 300u32);
}
