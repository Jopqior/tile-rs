// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

// Raw pointer creation and dereferencing
fn test_raw_ptr_from_ref() {
    let x = 42i32;
    let ptr: *const i32 = &x as *const i32;
    let val = unsafe { *ptr };
    assert_eq!(val, 42);
}

fn test_raw_ptr_mut() {
    let mut x = 10i32;
    let ptr: *mut i32 = &mut x as *mut i32;
    unsafe { *ptr = 20; }
    assert_eq!(x, 20);
}

// Pointer arithmetic with wrapping_add
fn test_ptr_offset() {
    let arr = [10i32, 20, 30, 40, 50];
    let base: *const i32 = &arr[0] as *const i32;
    unsafe {
        assert_eq!(*base, 10);
        assert_eq!(*base.wrapping_add(1), 20);
        assert_eq!(*base.wrapping_add(2), 30);
        assert_eq!(*base.wrapping_add(4), 50);
    }
}

fn test_ptr_mut_write() {
    let mut arr = [0i32; 4];
    let base: *mut i32 = &mut arr[0] as *mut i32;
    unsafe {
        *base = 100;
        *base.wrapping_add(1) = 200;
        *base.wrapping_add(2) = 300;
        *base.wrapping_add(3) = 400;
    }
    assert_eq!(arr[0], 100);
    assert_eq!(arr[1], 200);
    assert_eq!(arr[2], 300);
    assert_eq!(arr[3], 400);
}

// Cast between pointer types
fn test_ptr_cast() {
    let x = 0x12345678u32;
    let ptr: *const u32 = &x as *const u32;
    let byte_ptr: *const u8 = ptr as *const u8;
    unsafe {
        // Read first byte (little-endian: 0x78)
        let first_byte = *byte_ptr;
        assert_eq!(first_byte, 0x78u8);
    }
}

// Pointer to struct
struct Pair {
    a: i32,
    b: i32,
}

fn test_ptr_to_struct() {
    let p = Pair { a: 10, b: 20 };
    let ptr: *const Pair = &p as *const Pair;
    unsafe {
        assert_eq!((*ptr).a, 10);
        assert_eq!((*ptr).b, 20);
    }
}

fn test_ptr_mut_struct() {
    let mut p = Pair { a: 0, b: 0 };
    let ptr: *mut Pair = &mut p as *mut Pair;
    unsafe {
        (*ptr).a = 100;
        (*ptr).b = 200;
    }
    assert_eq!(p.a, 100);
    assert_eq!(p.b, 200);
}

// Null pointer creation
fn test_null_ptr() {
    let null: *const i32 = core::ptr::null();
    let null_mut: *mut i32 = core::ptr::null_mut();
    // Can create null pointers without crashing
    let addr = null as usize;
    assert_eq!(addr, 0usize);
    let addr_mut = null_mut as usize;
    assert_eq!(addr_mut, 0usize);
}

// Pointer to usize roundtrip
fn test_ptr_to_usize() {
    let x = 42i32;
    let ptr: *const i32 = &x as *const i32;
    let addr = ptr as usize;
    // Address should be non-zero (stack-allocated)
    assert!(addr != 0);
    // Roundtrip back
    let ptr2 = addr as *const i32;
    unsafe {
        assert_eq!(*ptr2, 42);
    }
}

// Array traversal via pointers
fn sum_via_ptr(arr: &[i32; 5]) -> i32 {
    let base: *const i32 = &arr[0] as *const i32;
    let mut sum = 0i32;
    let mut i = 0usize;
    loop {
        if i >= 5 { break; }
        unsafe {
            sum = sum + *base.wrapping_add(i);
        }
        i = i + 1;
    }
    sum
}

fn test_ptr_traversal() {
    let arr = [1i32, 2, 3, 4, 5];
    assert_eq!(sum_via_ptr(&arr), 15);
}
