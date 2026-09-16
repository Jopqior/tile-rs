// build-pass

#![feature(no_core)]

#![no_std]
#![no_core]

extern crate tile_std;

const LUT: [u8; 4] = [1, 1, 1, 1];

pub fn decode(i: u8) -> u8 {
    if i < 4 { LUT[i as usize] } else { 2 }
}
