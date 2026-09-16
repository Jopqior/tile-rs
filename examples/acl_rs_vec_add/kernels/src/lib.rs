#![feature(no_core)]

#![no_std]
#![no_core]

#[tile_std::tile_kernel]
pub fn add(x: *const u16, y: *const u16, z: *mut u16) {
    unsafe {
        let block_size = 16usize / tile_std::get_block_num();
        let start = tile_std::get_block_idx() * block_size;
        let mut i = start;
        loop {
            *z.wrapping_add(i) = *x.wrapping_add(i) + *y.wrapping_add(i);

            i = i + 1;
            if i == block_size + start {
                break;
            }
        }
    }
}
