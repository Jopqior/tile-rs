// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]
#![allow(unnecessary_transmutes)]

extern crate tile_std;
use tile_std::*;

fn test_size_of() {
    assert_eq!(core::mem::size_of::<u8>(), 1);
    assert_eq!(core::mem::size_of::<u16>(), 2);
    assert_eq!(core::mem::size_of::<u32>(), 4);
    assert_eq!(core::mem::size_of::<u64>(), 8);
    assert_eq!(core::mem::size_of::<i32>(), 4);
    assert_eq!(core::mem::size_of::<f32>(), 4);
    assert_eq!(core::mem::size_of::<bool>(), 1);
}

fn test_transmute() {
    // f32 to u32 bit pattern
    let x: f32 = 1.0;
    let bits: u32 = unsafe { core::mem::transmute(x) };
    assert_eq!(bits, 0x3F800000); // IEEE 754 for 1.0f32

    // u32 to f32
    let y: u32 = 0x40000000; // IEEE 754 for 2.0f32
    let f: f32 = unsafe { core::mem::transmute(y) };
    assert_eq!(f, 2.0);

    // i8 to u8
    let a: i8 = -1;
    let b: u8 = unsafe { core::mem::transmute(a) };
    assert_eq!(b, 255);
}

fn test_replace() {
    let mut x: i32 = 42;
    let old = core::mem::replace(&mut x, 100);
    assert_eq!(old, 42);
    assert_eq!(x, 100);

    let mut y: u8 = 0xFF;
    let old_y = core::mem::replace(&mut y, 0x00);
    assert_eq!(old_y, 0xFF);
    assert_eq!(y, 0x00);
}

fn test_swap() {
    let mut a: i32 = 10;
    let mut b: i32 = 20;
    core::mem::swap(&mut a, &mut b);
    assert_eq!(a, 20);
    assert_eq!(b, 10);
}

fn test_ptr_read() {
    let x: u32 = 0xDEADBEEF;
    let val = unsafe { core::ptr::read(&x as *const u32) };
    assert_eq!(val, 0xDEADBEEF);

    let arr: [u8; 4] = [1, 2, 3, 4];
    let first = unsafe { core::ptr::read(&arr[0] as *const u8) };
    assert_eq!(first, 1);
}

