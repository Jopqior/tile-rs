// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::*;

fn test_bitand() {
    let a: u32 = 0xFF00;
    let b: u32 = 0x0FF0;
    assert_eq!(a & b, 0x0F00);

    let x: i8 = 0b1100;
    let y: i8 = 0b1010;
    assert_eq!(x & y, 0b1000);

    assert_eq!(true & true, true);
    assert_eq!(true & false, false);
}

fn test_bitand_assign() {
    let mut a: u32 = 0xFF;
    a &= 0x0F;
    assert_eq!(a, 0x0F);
}

fn test_bitor() {
    let a: u32 = 0xF000;
    let b: u32 = 0x00F0;
    assert_eq!(a | b, 0xF0F0);

    assert_eq!(false | true, true);
    assert_eq!(false | false, false);
}

fn test_bitor_assign() {
    let mut a: u32 = 0xF0;
    a |= 0x0F;
    assert_eq!(a, 0xFF);
}

fn test_bitxor() {
    let a: u32 = 0xFF00;
    let b: u32 = 0xF0F0;
    assert_eq!(a ^ b, 0x0FF0);

    assert_eq!(true ^ true, false);
    assert_eq!(true ^ false, true);
}

fn test_bitxor_assign() {
    let mut a: u32 = 0xFF;
    a ^= 0x0F;
    assert_eq!(a, 0xF0);
}

fn test_shl() {
    let a: u32 = 1;
    assert_eq!(a << 4u32, 16);

    let b: u8 = 0b0001;
    assert_eq!(b << 3u32, 0b1000);

    // Cross-type shifts
    let c: u32 = 1;
    assert_eq!(c << 8u8, 256);
    assert_eq!(c << 16u16, 65536);
}

fn test_shl_assign() {
    let mut a: u32 = 1;
    a <<= 4u32;
    assert_eq!(a, 16);
}

fn test_shr() {
    let a: u32 = 256;
    assert_eq!(a >> 4u32, 16);

    // Signed right shift (arithmetic)
    let b: i32 = -16;
    let c = b >> 2i32;
    let _ = c; // sign-extends

    // Cross-type shifts
    let d: u32 = 256;
    assert_eq!(d >> 8u8, 1);
}

fn test_shr_assign() {
    let mut a: u32 = 256;
    a >>= 4u32;
    assert_eq!(a, 16);
}

fn test_not() {
    assert_eq!(!true, false);
    assert_eq!(!false, true);

    let a: u8 = 0;
    assert_eq!(!a, 0xFF);
}

fn test_combined_bitwise() {
    // Common pattern: set, clear, toggle, check bits
    let flags: u32 = 0;
    let mask: u32 = 1 << 3u32; // bit 3

    // Set bit
    let flags = flags | mask;
    assert_eq!(flags & mask, mask);

    // Toggle bit
    let flags = flags ^ mask;
    assert_eq!(flags & mask, 0);

    // Clear bit (set then clear)
    let flags = flags | mask;
    let flags = flags & !mask;
    assert_eq!(flags & mask, 0);
}
