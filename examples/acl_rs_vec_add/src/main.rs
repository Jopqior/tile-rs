use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use core::assert_eq;
use log::info;
use simple_logger::SimpleLogger;
use std::ffi::c_void;

unsafe extern "C" {
    fn add_do(
        dim: u32,
        stream: *mut c_void,
        x: *mut c_void,
        y: *mut c_void,
        z: *mut c_void,
    );
}

fn main() -> Result<()> {
    SimpleLogger::new()
        .init()
        .context("Failed to initialize logger")?;
    info!("Logger initialized successfully.");

    let block_dim: u32 = 2;

    info!("Initializing ACL...");
    let acl = Acl::new()?;
    info!("ACL initialized successfully.");

    info!("Initializing device...");
    let device = Device::new(&acl)?;
    info!("device initialized successfully.");

    info!("Creating ACL context and stream...");
    let context = AclContext::new(&device)?;
    let stream = AclStream::new(&context)?;

    // Read input data from files
    info!("Attempting to read vec X from 'test_data/input_x.bin'...");
    let x_host = common::read_buf_from_file::<i16>("test_data/input_x.bin");
    info!("Read {} elems for vec X", x_host.len());

    info!("Attempting to read vec Y from 'test_data/input_y.bin'...");
    let y_host = common::read_buf_from_file::<i16>("test_data/input_y.bin");
    info!("Read {} elems for vec Y", y_host.len());

    info!("Copying vec X from host to device...");
    let mut x_device =
        DeviceBuffer::from_slice_with_policy(x_host.as_slice(), AclrtMemMallocPolicy::HugeFirst)?;
    info!("Copying vec Y from host to device...");
    let mut y_device =
        DeviceBuffer::from_slice_with_policy(y_host.as_slice(), AclrtMemMallocPolicy::HugeFirst)?;
    let mut z_device = unsafe {
        DeviceBuffer::<i16>::uninitialized_with_policy(
            x_host.len(),
            AclrtMemMallocPolicy::HugeFirst,
        )?
    };

    info!("Running kernel...");
    unsafe {
        add_do(
            block_dim,
            stream.to_raw(),
            x_device.as_mut_ptr() as *mut _,
            y_device.as_mut_ptr() as *mut _,
            z_device.as_mut_ptr() as *mut _,
        );
    }

    info!("Synchronizing the stream...");
    stream.synchronize()?;

    let res = z_device.to_host()?;
    info!("Copying vec Z from device to host...");

    info!("Comparing output...");
    for (idx, elem) in res.iter().enumerate() {
        assert_eq!(*elem, x_host[idx] + y_host[idx]);
    }
    info!("Succeed");

    Ok(())
}
