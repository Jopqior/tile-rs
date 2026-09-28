// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;
use tile_std::cmp::Ordering;
use tile_std::clone::Clone;
use tile_std::cmp::Ord;

fn test_clone() {
    let a = (1, "2");
    let b = a.clone();
    assert_eq!(a, b);
}

fn test_partial_eq() {
    let (small, big) = ((1, 2, 3), (3, 2, 1));
    assert_eq!(small, small);
    assert_eq!(big, big);
    assert_ne!(small, big);
    assert_ne!(big, small);
}

fn test_partial_ord() {
    let (small, big) = ((1, 2, 3), (3, 2, 1));

    assert!(small < big);
    assert!(!(small < small));
    assert!(!(big < small));
    assert!(!(big < big));

    assert!(small <= small);
    assert!(big <= big);

    assert!(big > small);
    assert!(small >= small);
    assert!(big >= small);
    assert!(big >= big);

    assert!(!((1.0f32, 2.0f32) < (f32::NAN, 3.0)));
    assert!(!((1.0f32, 2.0f32) <= (f32::NAN, 3.0)));
    assert!(!((1.0f32, 2.0f32) > (f32::NAN, 3.0)));
    assert!(!((1.0f32, 2.0f32) >= (f32::NAN, 3.0)));
    assert!((1.0f32, 2.0f32) < (2.0, f32::NAN));
    assert!(!((2.0f32, 2.0f32) < (2.0, f32::NAN)));
}

fn test_ord() {
    let (small, big) = ((1, 2, 3), (3, 2, 1));
    assert_eq!(small.cmp(&small), Ordering::Equal);
    assert_eq!(big.cmp(&big), Ordering::Equal);
    assert_eq!(small.cmp(&big), Ordering::Less);
    assert_eq!(big.cmp(&small), Ordering::Greater);
}

fn test_show() {
    let s = format!("{:?}", (1,));
    assert_eq!(s, "(1,)");
    let s = format!("{:?}", (1, true));
    assert_eq!(s, "(1, true)");
    let s = format!("{:?}", (1, "hi", true));
    assert_eq!(s, "(1, \"hi\", true)");
}
