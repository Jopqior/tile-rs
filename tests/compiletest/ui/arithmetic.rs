// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

fn test_sub() {
    let a: i32 = 10;
    let b: i32 = 3;
    let c = a - b;
    assert_eq!(c, 7);

    let x: u64 = 100;
    let y: u64 = 42;
    assert_eq!(x - y, 58);

    let f: f32 = 3.5;
    let g: f32 = 1.5;
    let _h = f - g;
}

fn test_sub_assign() {
    let mut a: i32 = 20;
    a -= 5;
    assert_eq!(a, 15);

    let mut f: f32 = 10.0;
    f -= 2.5;
}

fn test_rem() {
    let a: i32 = 17;
    let b: i32 = 5;
    assert_eq!(a % b, 2);

    let x: u32 = 10;
    let y: u32 = 3;
    assert_eq!(x % y, 1);
}

fn test_rem_signed() {
    let a: i32 = -17;
    let b: i32 = 5;
    assert_eq!(a % b, -2);

    let c: i64 = -10;
    let d: i64 = 3;
    assert_eq!(c % d, -1);
}

fn test_rem_assign() {
    let mut a: u32 = 17;
    a %= 5;
    assert_eq!(a, 2);
}

fn test_mul_assign() {
    let mut a: i32 = 6;
    a *= 7;
    assert_eq!(a, 42);

    let mut f: f32 = 2.0;
    f *= 3.0;
}

fn test_div_assign() {
    let mut a: i32 = 42;
    a /= 7;
    assert_eq!(a, 6);

    let mut f: f32 = 10.0;
    f /= 2.0;
}

fn test_neg() {
    let a: i32 = 5;
    assert_eq!(-a, -5);

    let b: i32 = -3;
    assert_eq!(-b, 3);
}

fn test_mixed_arithmetic() {
    let a: i32 = 10;
    let b: i32 = 3;
    let c = a + b;      // Add
    let d = c - b;      // Sub
    let e = d * b;       // Mul
    let f = e / b;       // Div
    let g = f % b;       // Rem
    assert_eq!(g, 1);    // 10 % 3 = 1
}
