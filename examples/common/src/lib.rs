use ascend_rs::prelude::*;

use std::fs;
use std::{io::Read, path::Path};

pub fn read_buf_from_file<T: DeviceSend>(path: impl AsRef<Path>) -> LockedBuffer<T> {
    let mut file = fs::File::open(&path).expect("no file found");
    let metadata = fs::metadata(&path).expect("unable to open file metadata");
    let mut buffer = unsafe {
        LockedBuffer::<u8>::uninitialized(metadata.len() as usize).expect("unable to crate buffer")
    };
    file.read(&mut buffer).expect("buffer overflow");
    unsafe {
        let len = buffer.len() / std::mem::size_of::<T>();
        let ptr = buffer.as_mut_ptr() as *mut T;

        std::mem::forget(buffer);
        LockedBuffer::from_raw_parts(ptr, len)
    }
}
