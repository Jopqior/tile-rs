// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

fn test_count_ones() {
    let x: u32 = 0b1010_1010;
    assert_eq!(x.count_ones(), 4);

    let y: u32 = 0;
    assert_eq!(y.count_ones(), 0);

    let z: u32 = u32::MAX;
    assert_eq!(z.count_ones(), 32);

    let a: u8 = 0b1111_0000;
    assert_eq!(a.count_ones(), 4);
}

fn test_count_zeros() {
    let x: u32 = 0b1010_1010;
    assert_eq!(x.count_zeros(), 28);

    let y: u8 = 0xFF;
    assert_eq!(y.count_zeros(), 0);
}

fn test_leading_zeros() {
    let x: u32 = 1;
    assert_eq!(x.leading_zeros(), 31);

    let y: u32 = 0;
    assert_eq!(y.leading_zeros(), 32);

    let z: u32 = u32::MAX;
    assert_eq!(z.leading_zeros(), 0);

    let a: u8 = 0b0000_1000;
    assert_eq!(a.leading_zeros(), 4);
}

fn test_trailing_zeros() {
    let x: u32 = 0b1000;
    assert_eq!(x.trailing_zeros(), 3);

    let y: u32 = 1;
    assert_eq!(y.trailing_zeros(), 0);

    let z: u32 = 0;
    assert_eq!(z.trailing_zeros(), 32);

    let a: u8 = 0b0011_0000;
    assert_eq!(a.trailing_zeros(), 4);
}

fn test_swap_bytes() {
    let x: u32 = 0x12345678;
    assert_eq!(x.swap_bytes(), 0x78563412);

    let y: u16 = 0xAABB;
    assert_eq!(y.swap_bytes(), 0xBBAA);

    let z: u8 = 0xAB;
    assert_eq!(z.swap_bytes(), 0xAB); // single byte, no change
}

fn test_reverse_bits() {
    let x: u8 = 0b1010_0000;
    assert_eq!(x.reverse_bits(), 0b0000_0101);

    let y: u16 = 0b1000_0000_0000_0000;
    assert_eq!(y.reverse_bits(), 1);
}

fn test_rotate_left() {
    let x: u32 = 0x10000002;
    assert_eq!(x.rotate_left(4), 0x00000021);

    let y: u8 = 0b1000_0001;
    assert_eq!(y.rotate_left(1), 0b0000_0011);
}

fn test_rotate_right() {
    let x: u32 = 0x00000021;
    assert_eq!(x.rotate_right(4), 0x10000002);

    let y: u8 = 0b0000_0011;
    assert_eq!(y.rotate_right(1), 0b1000_0001);
}

fn test_signed_bit_ops() {
    let x: i32 = -1;
    assert_eq!(x.count_ones(), 32);
    assert_eq!(x.leading_zeros(), 0);
    assert_eq!(x.trailing_zeros(), 0);

    let y: i32 = 0;
    assert_eq!(y.count_ones(), 0);
    assert_eq!(y.leading_zeros(), 32);

    let z: i32 = 1;
    assert_eq!(z.leading_zeros(), 31);
    assert_eq!(z.trailing_zeros(), 0);
}

fn test_signed_swap_reverse() {
    let x: i16 = 0x1234;
    assert_eq!(x.swap_bytes(), 0x3412);

    let y: i32 = 1;
    assert_eq!(y.reverse_bits(), i32::MIN);
}

fn test_signed_rotate() {
    let x: i32 = 0x10000002;
    assert_eq!(x.rotate_left(4), 0x00000021);
    assert_eq!(x.rotate_right(28), 0x00000021);
}
