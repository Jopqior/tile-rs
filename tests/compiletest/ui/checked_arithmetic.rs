// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Test checked add (unsigned)
fn test_checked_add_u32() {
    let a: u32 = 100;
    let b: u32 = 200;
    match a.checked_add(b) {
        Some(val) => assert_eq!(val, 300),
        None => panic!("unexpected overflow"),
    }
}

fn test_checked_add_u32_overflow() {
    let a: u32 = u32::MAX;
    let b: u32 = 1;
    match a.checked_add(b) {
        Some(_) => panic!("expected overflow"),
        None => {} // correct
    }
}

// Test checked add (signed)
fn test_checked_add_i32() {
    let a: i32 = 100;
    let b: i32 = -50;
    match a.checked_add(b) {
        Some(val) => assert_eq!(val, 50),
        None => panic!("unexpected overflow"),
    }
}

fn test_checked_add_i32_overflow() {
    let a: i32 = i32::MAX;
    let b: i32 = 1;
    match a.checked_add(b) {
        Some(_) => panic!("expected overflow"),
        None => {} // correct
    }
}

fn test_checked_add_i32_underflow() {
    let a: i32 = i32::MIN;
    let b: i32 = -1;
    match a.checked_add(b) {
        Some(_) => panic!("expected overflow"),
        None => {} // correct
    }
}

// Test checked sub (unsigned)
fn test_checked_sub_u32() {
    let a: u32 = 300;
    let b: u32 = 100;
    match a.checked_sub(b) {
        Some(val) => assert_eq!(val, 200),
        None => panic!("unexpected overflow"),
    }
}

fn test_checked_sub_u32_underflow() {
    let a: u32 = 0;
    let b: u32 = 1;
    match a.checked_sub(b) {
        Some(_) => panic!("expected underflow"),
        None => {} // correct
    }
}

// Test checked sub (signed)
fn test_checked_sub_i32() {
    let a: i32 = -100;
    let b: i32 = -200;
    match a.checked_sub(b) {
        Some(val) => assert_eq!(val, 100),
        None => panic!("unexpected overflow"),
    }
}

fn test_checked_sub_i32_overflow() {
    let a: i32 = i32::MIN;
    let b: i32 = 1;
    match a.checked_sub(b) {
        Some(_) => panic!("expected overflow"),
        None => {} // correct
    }
}

// Test checked mul (unsigned)
fn test_checked_mul_u32() {
    let a: u32 = 100;
    let b: u32 = 200;
    match a.checked_mul(b) {
        Some(val) => assert_eq!(val, 20000),
        None => panic!("unexpected overflow"),
    }
}

fn test_checked_mul_u32_overflow() {
    let a: u32 = u32::MAX;
    let b: u32 = 2;
    match a.checked_mul(b) {
        Some(_) => panic!("expected overflow"),
        None => {} // correct
    }
}

// Test checked mul (signed)
fn test_checked_mul_i32() {
    let a: i32 = -10;
    let b: i32 = 20;
    match a.checked_mul(b) {
        Some(val) => assert_eq!(val, -200),
        None => panic!("unexpected overflow"),
    }
}

fn test_checked_mul_i32_overflow() {
    let a: i32 = i32::MAX;
    let b: i32 = 2;
    match a.checked_mul(b) {
        Some(_) => panic!("expected overflow"),
        None => {} // correct
    }
}

// Test wrapping operations (use same underlying ops)
fn test_wrapping_add() {
    let a: u32 = u32::MAX;
    let b: u32 = 1;
    assert_eq!(a.wrapping_add(b), 0);
}

fn test_wrapping_sub() {
    let a: u32 = 0;
    let b: u32 = 1;
    assert_eq!(a.wrapping_sub(b), u32::MAX);
}

// Test overflowing operations
fn test_overflowing_add() {
    let (val, overflow) = 200u32.overflowing_add(100);
    assert_eq!(val, 300);
    assert!(!overflow);

    let (val, overflow) = u32::MAX.overflowing_add(1);
    assert_eq!(val, 0);
    assert!(overflow);
}

fn test_overflowing_sub() {
    let (val, overflow) = 300u32.overflowing_sub(100);
    assert_eq!(val, 200);
    assert!(!overflow);

    let (val, overflow) = 0u32.overflowing_sub(1);
    assert_eq!(val, u32::MAX);
    assert!(overflow);
}
