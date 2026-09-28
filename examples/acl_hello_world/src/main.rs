use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;

use core::ffi::c_void;
use std::error::Error;

unsafe extern "C" {
    fn hello_world_do(dim: u32, stream: *mut c_void);
}

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;

    // Initialize ACL and log the result
    info!("Initializing ACL...");
    let acl = Acl::new()?;
    info!("ACL initialized successfully.");

    // Device ID configuration
    info!("Initializing device...");
    let device = Device::new(&acl)?;
    info!("device initialized successfully.");

    info!("Creating ACL context and stream...");
    let context = AclContext::new(&device)?;
    let stream = AclStream::new(&context)?;

    info!("Running kernel...");
    unsafe {
        hello_world_do(8, stream.to_raw());
    }

    // Synchronize stream
    info!("Synchronizing stream...");
    stream.synchronize()?;
    info!("Stream synchronized successfully.");

    // Context and stream are dropped here, destroying them.
    info!("Context and stream destroyed successfully.");

    // ACL is dropped here, resetting device and finalizing.
    info!("ACL finalized successfully.");

    Ok(())
}
