// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Test tuple equality
fn test_tuple_eq() {
    let a = (1i32, 2i32);
    let b = (1i32, 2i32);
    let c = (1i32, 3i32);
    assert!(a == b);
    assert!(a != c);
}

// Test tuple ordering - lt
fn test_tuple_lt() {
    assert!((1i32, 2i32) < (1i32, 3i32));
    assert!((1i32, 2i32) < (2i32, 0i32));
    assert!(!((1i32, 2i32) < (1i32, 2i32)));
    assert!(!((2i32, 0i32) < (1i32, 3i32)));
}

// Test tuple ordering - le
fn test_tuple_le() {
    assert!((1i32, 2i32) <= (1i32, 3i32));
    assert!((1i32, 2i32) <= (1i32, 2i32));
    assert!(!((1i32, 3i32) <= (1i32, 2i32)));
}

// Test tuple ordering - gt
fn test_tuple_gt() {
    assert!((1i32, 3i32) > (1i32, 2i32));
    assert!((2i32, 0i32) > (1i32, 3i32));
    assert!(!((1i32, 2i32) > (1i32, 2i32)));
    assert!(!((1i32, 2i32) > (2i32, 0i32)));
}

// Test tuple ordering - ge
fn test_tuple_ge() {
    assert!((1i32, 3i32) >= (1i32, 2i32));
    assert!((1i32, 2i32) >= (1i32, 2i32));
    assert!(!((1i32, 2i32) >= (1i32, 3i32)));
}

// Test single-element tuple
fn test_single_tuple() {
    assert!((1i32,) < (2i32,));
    assert!(!((2i32,) < (1i32,)));
    assert!((1i32,) <= (1i32,));
}

// Test triple tuple
fn test_triple_tuple() {
    assert!((1i32, 2i32, 3i32) < (1i32, 2i32, 4i32));
    assert!((1i32, 2i32, 3i32) < (1i32, 3i32, 0i32));
    assert!(!((1i32, 2i32, 3i32) < (1i32, 2i32, 3i32)));
}

// Test tuple comparison used in sorting-like logic
fn test_tuple_min_max() {
    let a = (1i32, 10i32);
    let b = (1i32, 5i32);

    let min = if a < b { a } else { b };
    assert_eq!(min.0, 1);
    assert_eq!(min.1, 5);

    let max = if a > b { a } else { b };
    assert_eq!(max.0, 1);
    assert_eq!(max.1, 10);
}
