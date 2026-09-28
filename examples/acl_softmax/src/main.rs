// =============================================================================
// Host Application: Launches the softmax NPU kernel and verifies results
// =============================================================================
//
// Execution flow:
//   1. Initialize ACL runtime and select device
//   2. Create context + stream
//   3. Load f32 input data and u32 length from binary files
//   4. Allocate device memory and copy inputs host → device
//   5. Load and launch the softmax kernel (single block)
//   6. Synchronize, copy results device → host
//   7. Verify against CPU reference: sum ≈ 1.0 and element-wise match

use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use std::ffi::c_void;

/// CPU reference implementation of numerically stable softmax
fn cpu_softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum).collect()
}

fn main() -> Result<()> {
    SimpleLogger::new().init()?;

    // -- Step 1: Initialize ACL and select device 0 --------------------------
    let acl = Acl::new()?;
    let device = Device::new(&acl)?;
    info!("Device initialized: {}", device.descriptor());

    // -- Step 2: Create context and stream -----------------------------------
    let context = AclContext::new(&device)?;
    let stream = AclStream::new(&context)?;

    // -- Step 3: Load input data ---------------------------------------------
    let input_host = common::read_buf_from_file::<f32>("test_data/input.bin");
    let len_host = common::read_buf_from_file::<u32>("test_data/len.bin");
    info!(
        "Loaded {} elements, len parameter = {}",
        input_host.len(),
        len_host[0]
    );

    // -- Step 4: Allocate device memory and copy data ------------------------
    let mut input_device = DeviceBuffer::from_slice_with_policy(
        input_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )?;
    let mut len_device =
        DeviceBuffer::from_slice_with_policy(len_host.as_slice(), AclrtMemMallocPolicy::HugeFirst)?;
    let mut output_device = unsafe {
        DeviceBuffer::<f32>::uninitialized_with_policy(
            input_host.len(),
            AclrtMemMallocPolicy::HugeFirst,
        )?
    };

    // -- Step 5: Load and launch the softmax kernel --------------------------
    // Launch via the generated `softmax_do` extern "C" wrapper (the AscendC
    // launch entry), dlsym'd from kernel.o. KernelLoader::launch (rtKernelLaunch
    // on the raw stub) does not set up the AscendC launch context and fails ACL
    // "Parameter validation failed" for these kernels.
    type KernelDoFn = unsafe extern "C" fn(u32, *mut c_void, *mut c_void, *mut c_void, *mut c_void);
    let kernel_path = concat!(env!("OUT_DIR"), "/kernel.o");
    let kernel_lib =
        unsafe { libloading::Library::new(kernel_path) }.context("dlopen kernel.o")?;
    unsafe {
        let softmax_do: KernelDoFn =
            *kernel_lib.get(b"softmax_do\0").context("dlsym softmax_do")?;
        // Single block — all elements processed by one block.
        softmax_do(
            1,
            stream.to_raw(),
            input_device.as_mut_ptr() as *mut c_void,
            output_device.as_mut_ptr() as *mut c_void,
            len_device.as_mut_ptr() as *mut c_void,
        );
    }

    // -- Step 6: Synchronize and copy results --------------------------------
    stream.synchronize()?;
    let result = output_device.to_host()?;

    // -- Step 7: Verify results ----------------------------------------------
    let expected = cpu_softmax(input_host.as_slice());

    // Check sum ≈ 1.0
    let sum: f32 = result.iter().sum();
    info!("Output sum: {} (expected ≈ 1.0)", sum);
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax output sum {} is not close to 1.0",
        sum
    );

    // Element-wise comparison
    info!("Verifying {} results...", result.len());
    for (idx, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at index {}: got {}, expected {}",
            idx,
            got,
            exp
        );
    }
    info!("All results correct!");

    Ok(())
}
