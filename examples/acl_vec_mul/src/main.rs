// =============================================================================
// Host Application: Launches the NPU kernel and verifies results
// =============================================================================
//
// End-to-end execution flow:
//
//   1. Initialize ACL runtime and select device
//   2. Create a context + stream for asynchronous execution
//   3. Load input data from binary files into page-locked host memory
//   4. Allocate device memory and copy inputs host -> device
//   5. Load the compiled kernel binary (`kernel.o` from build.rs)
//   6. Launch the kernel with specified block dimensions
//   7. Synchronize the stream (wait for kernel completion)
//   8. Copy results device -> host
//   9. Verify output against expected values
//  10. Resources are freed automatically via RAII (Drop impls)

use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use std::ffi::c_void;

fn main() -> Result<()> {
    SimpleLogger::new().init()?;

    // -- Step 1: Initialize ACL and select device 0 --------------------------
    let acl = Acl::new()?;
    let device = Device::new(&acl)?;
    info!("Device initialized: {}", device.descriptor());

    // -- Step 2: Create context and stream -----------------------------------
    // A context binds the current thread to a device.
    // A stream provides ordered, asynchronous execution of kernels and memcpys.
    let context = AclContext::new(&device)?;
    let stream = AclStream::new(&context)?;

    // -- Step 3: Load input data ---------------------------------------------
    // `read_buf_from_file` reads raw binary into page-locked (pinned) host
    // memory, which enables fast DMA transfers to device memory.
    let x_host = common::read_buf_from_file::<u16>("test_data/input_x.bin");
    let y_host = common::read_buf_from_file::<u16>("test_data/input_y.bin");
    info!("Loaded {} elements per input vector", x_host.len());

    // -- Step 4: Allocate device memory and copy data ------------------------
    // HugeFirst policy prefers large-page allocations for better TLB behavior.
    let mut x_device =
        DeviceBuffer::from_slice_with_policy(x_host.as_slice(), AclrtMemMallocPolicy::HugeFirst)?;
    let mut y_device =
        DeviceBuffer::from_slice_with_policy(y_host.as_slice(), AclrtMemMallocPolicy::HugeFirst)?;
    let mut z_device = unsafe {
        DeviceBuffer::<u16>::uninitialized_with_policy(
            x_host.len(),
            AclrtMemMallocPolicy::HugeFirst,
        )?
    };

    // -- Step 5: Load the compiled kernel binary -----------------------------
    // KernelLoader::new() reads `$OUT_DIR/kernel.o` (produced by build.rs)
    // and registers the binary with the Ascend runtime via rtDevBinaryRegister.
    //
    // get_kernel("mul") registers the "mul" function symbol via
    // rtFunctionRegister so the runtime can dispatch it.
    // -- Step 6: Launch the kernel ---------------------------------------
    // Launch via the generated `mul_do` extern "C" wrapper (the AscendC launch
    // entry), dlsym'd from kernel.o. KernelLoader::launch (rtKernelLaunch on the
    // raw stub) does not set up the AscendC launch context and fails ACL
    // "Parameter validation failed" for these kernels.
    // block_dim = 2: get_block_num()==2; block 0 does [0..8), block 1 [8..16).
    type KernelDoFn = unsafe extern "C" fn(u32, *mut c_void, *mut c_void, *mut c_void, *mut c_void);
    let kernel_path = concat!(env!("OUT_DIR"), "/kernel.o");
    let kernel_lib =
        unsafe { libloading::Library::new(kernel_path) }.context("dlopen kernel.o")?;
    unsafe {
        let mul_do: KernelDoFn = *kernel_lib.get(b"mul_do\0").context("dlsym mul_do")?;
        let block_dim: u32 = 2;
        mul_do(
            block_dim,
            stream.to_raw(),
            x_device.as_mut_ptr() as *mut c_void,
            y_device.as_mut_ptr() as *mut c_void,
            z_device.as_mut_ptr() as *mut c_void,
        );
    }

    // -- Step 7: Synchronize -------------------------------------------------
    // Block until all operations on this stream have completed.
    stream.synchronize()?;

    // -- Step 8: Copy result back to host ------------------------------------
    let res = z_device.to_host()?;

    // -- Step 9: Verify results ----------------------------------------------
    info!("Verifying {} results...", res.len());
    for (idx, elem) in res.iter().enumerate() {
        let expected = x_host[idx].wrapping_mul(y_host[idx]);
        assert_eq!(
            *elem, expected,
            "mismatch at index {}: got {}, expected {} ({}*{})",
            idx, elem, expected, x_host[idx], y_host[idx]
        );
    }
    info!("All results correct!");

    Ok(())
}
