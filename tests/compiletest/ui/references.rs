// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Basic immutable reference
fn test_immutable_ref() {
    let x = 42i32;
    let r = &x;
    assert_eq!(*r, 42);
}

// Mutable reference
fn test_mutable_ref() {
    let mut x = 10i32;
    let r = &mut x;
    *r = 20;
    assert_eq!(x, 20);
}

// Reference to struct field
struct Point {
    x: i32,
    y: i32,
}

fn test_struct_ref() {
    let p = Point { x: 3, y: 4 };
    let rx = &p.x;
    let ry = &p.y;
    assert_eq!(*rx, 3);
    assert_eq!(*ry, 4);
}

fn test_struct_mut_ref() {
    let mut p = Point { x: 0, y: 0 };
    {
        let r = &mut p;
        r.x = 10;
        r.y = 20;
    }
    assert_eq!(p.x, 10);
    assert_eq!(p.y, 20);
}

// Pass by reference
fn add_ref(a: &i32, b: &i32) -> i32 {
    *a + *b
}

fn test_pass_by_ref() {
    let a = 3i32;
    let b = 4i32;
    assert_eq!(add_ref(&a, &b), 7);
}

// Modify through mutable reference
fn increment(val: &mut i32) {
    *val = *val + 1;
}

fn test_pass_by_mut_ref() {
    let mut x = 10i32;
    increment(&mut x);
    increment(&mut x);
    increment(&mut x);
    assert_eq!(x, 13);
}

// Reference to array element
fn test_array_ref() {
    let arr = [10i32, 20, 30, 40];
    let r = &arr[2];
    assert_eq!(*r, 30);
}

fn test_array_mut_ref() {
    let mut arr = [1i32, 2, 3];
    let r = &mut arr[1];
    *r = 42;
    assert_eq!(arr[1], 42);
}

// Reborrowing
fn test_reborrow() {
    let mut x = 5i32;
    let r1 = &mut x;
    *r1 = 10;
    // Implicit reborrow
    let r2 = &*r1;
    assert_eq!(*r2, 10);
}

// Reference in struct
struct Wrapper<'a> {
    value: &'a i32,
}

fn test_ref_in_struct() {
    let x = 100i32;
    let w = Wrapper { value: &x };
    assert_eq!(*w.value, 100);
}

// Multiple immutable references
fn test_multiple_refs() {
    let x = 42i32;
    let r1 = &x;
    let r2 = &x;
    let r3 = &x;
    assert_eq!(*r1 + *r2 + *r3, 126);
}
