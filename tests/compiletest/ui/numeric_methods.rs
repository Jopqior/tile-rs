// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// ---- Unsigned integer methods ----

fn test_uint_wrapping() {
    let a: u8 = 250;
    let b = a.wrapping_add(10); // wraps around
    let _ = b;

    let c: u32 = 5;
    let d = c.wrapping_sub(10); // wraps around
    let _ = d;

    let e: u64 = u64::MAX;
    let f = e.wrapping_mul(2);
    let _ = f;
}

fn test_uint_saturating() {
    let a: u8 = 250;
    let b = a.saturating_add(10);
    let _ = b; // should be 255

    let c: u32 = 5;
    let d = c.saturating_sub(10);
    assert_eq!(d, 0);
}

fn test_uint_checked() {
    let a: u8 = 200;
    let b = a.checked_add(100);
    assert!(b.is_none()); // overflow

    let c: u8 = 100;
    let d = c.checked_add(100);
    assert!(d.is_some());

    let e: u32 = 5;
    let f = e.checked_sub(10);
    assert!(f.is_none()); // underflow

    let g: u32 = 10;
    let h = g.checked_sub(5);
    assert_eq!(h, Some(5));

    let i: u8 = 200;
    let j = i.checked_mul(2);
    assert!(j.is_none()); // overflow
}

fn test_uint_min_max() {
    assert_eq!(3u32.min(5), 3);
    assert_eq!(3u32.max(5), 5);
    assert_eq!(5u32.min(3), 3);
    assert_eq!(5u32.max(3), 5);
}

fn test_uint_pow() {
    assert_eq!(2u32.pow(10), 1024);
    assert_eq!(3u32.pow(4), 81);
    assert_eq!(10u64.pow(0), 1);
    assert_eq!(7u32.pow(1), 7);
}

fn test_uint_minmax_constants() {
    assert_eq!(u8::MIN, 0);
    assert_eq!(u8::MAX, 255);
    assert_eq!(u32::MIN, 0);
}

// ---- Signed integer methods ----

fn test_sint_wrapping() {
    let a: i32 = i32::MAX;
    let b = a.wrapping_add(1);
    let _ = b; // wraps to i32::MIN

    let c: i8 = -128;
    let d = c.wrapping_sub(1);
    let _ = d; // wraps to 127

    let e: i32 = -5;
    let f = e.wrapping_neg();
    assert_eq!(f, 5);
}

fn test_sint_abs() {
    assert_eq!((-5i32).abs(), 5);
    assert_eq!(5i32.abs(), 5);
    assert_eq!(0i32.abs(), 0);
    assert_eq!((-100i64).abs(), 100);
}

fn test_sint_min_max() {
    assert_eq!((-3i32).min(5), -3);
    assert_eq!((-3i32).max(5), 5);
}

// ---- Float methods ----

fn test_float_nan() {
    let nan32: f32 = f32::NAN;
    assert!(nan32.is_nan());
    assert!(!1.0f32.is_nan());
    assert!(!0.0f32.is_nan());
}

fn test_float_infinite() {
    let inf: f32 = f32::INFINITY;
    let neg_inf: f32 = f32::NEG_INFINITY;

    assert!(inf.is_infinite());
    assert!(neg_inf.is_infinite());
    assert!(!1.0f32.is_infinite());
    assert!(!f32::NAN.is_infinite());
}

fn test_float_finite() {
    assert!(1.0f32.is_finite());
    assert!(0.0f32.is_finite());
    assert!(!f32::INFINITY.is_finite());
    assert!(!f32::NAN.is_finite());
}

fn test_float_abs() {
    assert_eq!((-2.5f32).abs(), 2.5);
    assert_eq!(3.5f32.abs(), 3.5);
    assert_eq!(0.0f32.abs(), 0.0);
}

fn test_float_min_max() {
    assert_eq!(1.0f32.min(2.0), 1.0);
    assert_eq!(1.0f32.max(2.0), 2.0);
    assert_eq!(2.0f32.min(1.0), 1.0);
    assert_eq!(2.0f32.max(1.0), 2.0);
}

fn test_float_clamp() {
    assert_eq!(1.5f32.clamp(0.0, 1.0), 1.0);
    assert_eq!((-0.5f32).clamp(0.0, 1.0), 0.0);
    assert_eq!(0.5f32.clamp(0.0, 1.0), 0.5);
}

fn test_float_constants() {
    let _nan32 = f32::NAN;
    let _inf32 = f32::INFINITY;
    let _neg_inf32 = f32::NEG_INFINITY;
}
