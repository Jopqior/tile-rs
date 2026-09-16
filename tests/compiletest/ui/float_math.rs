// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

fn test_floor() {
    let x: f32 = 3.7;
    assert_eq!(x.floor(), 3.0);

    let y: f32 = -2.3;
    assert_eq!(y.floor(), -3.0);

    let z: f32 = 5.0;
    assert_eq!(z.floor(), 5.0);
}

fn test_ceil() {
    let x: f32 = 3.2;
    assert_eq!(x.ceil(), 4.0);

    let y: f32 = -2.7;
    assert_eq!(y.ceil(), -2.0);

    let z: f32 = 5.0;
    assert_eq!(z.ceil(), 5.0);
}

fn test_round() {
    let x: f32 = 3.5;
    assert_eq!(x.round(), 4.0);

    let y: f32 = 3.4;
    assert_eq!(y.round(), 3.0);

    let z: f32 = -2.5;
    assert_eq!(z.round(), -3.0);
}

fn test_trunc() {
    let x: f32 = 3.9;
    assert_eq!(x.trunc(), 3.0);

    let y: f32 = -3.9;
    assert_eq!(y.trunc(), -3.0);
}

fn test_copysign() {
    let x: f32 = 5.0;
    assert_eq!(x.copysign(-1.0), -5.0);
    assert_eq!(x.copysign(1.0), 5.0);

    let y: f32 = -5.0;
    assert_eq!(y.copysign(1.0), 5.0);
    assert_eq!(y.copysign(-1.0), -5.0);
}

fn test_mul_add() {
    // fused multiply-add: self * a + b
    let x: f32 = 2.0;
    let result = x.mul_add(3.0, 4.0);
    assert_eq!(result, 10.0);

    let y: f32 = -1.0;
    let result2 = y.mul_add(5.0, 10.0);
    assert_eq!(result2, 5.0);
}

fn test_float_classify() {
    assert!(f32::NAN.is_nan());
    assert!(!f32::NAN.is_finite());
    assert!(f32::INFINITY.is_infinite());
    assert!(f32::NEG_INFINITY.is_infinite());
    assert!(!f32::INFINITY.is_finite());

    let x: f32 = 1.0;
    assert!(x.is_finite());
    assert!(!x.is_nan());
    assert!(!x.is_infinite());
}

fn test_float_min_max() {
    let a: f32 = 3.0;
    let b: f32 = 5.0;
    assert_eq!(a.min(b), 3.0);
    assert_eq!(a.max(b), 5.0);

    let c: f32 = -1.0;
    let d: f32 = 1.0;
    assert_eq!(c.min(d), -1.0);
    assert_eq!(c.max(d), 1.0);
}

fn test_float_abs() {
    let x: f32 = -5.0;
    assert_eq!(x.abs(), 5.0);

    let y: f32 = 5.0;
    assert_eq!(y.abs(), 5.0);

    let z: f32 = 0.0;
    assert_eq!(z.abs(), 0.0);
}

fn test_float_clamp() {
    let x: f32 = 10.0;
    assert_eq!(x.clamp(0.0, 5.0), 5.0);

    let y: f32 = -10.0;
    assert_eq!(y.clamp(0.0, 5.0), 0.0);

    let z: f32 = 3.0;
    assert_eq!(z.clamp(0.0, 5.0), 3.0);
}

fn test_exp_ln_sqrt() {
    // Basic exp/ln/sqrt (already tested elsewhere, but verify combo)
    let x: f32 = 1.0;
    let e = x.exp(); // e^1 ≈ 2.718

    let y: f32 = 4.0;
    assert_eq!(y.sqrt(), 2.0);

    let z: f32 = 1.0;
    assert_eq!(z.sqrt(), 1.0);
}
