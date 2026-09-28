// =============================================================================
// Host Application: Tile-API Softmax Test
// =============================================================================
//
// Tests the tile_softmax kernel written against the tile_std::tile API.
// The kernel uses PTO (Programmable Tile Operations) codegen path.
//
// Test: 1D softmax over 1024 f32 elements (ROWS=1, COLS=1024, 1 block).
// Verifies max error < 1e-5 and sum ≈ 1.0 against a CPU reference.

use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use std::ffi::c_void;

/// CPU reference: row-wise softmax (numerically stable).
fn softmax_cpu(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * cols];
    for i in 0..rows {
        let row = &input[i * cols..(i + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        for j in 0..cols {
            output[i * cols + j] = exp_vals[j] / sum;
        }
    }
    output
}

fn main() -> Result<()> {
    SimpleLogger::new().env().init().ok();

    // Select which kernel entry to exercise. Both have identical C ABI
    // and semantics; `_safe` is the view-based variant that serves as
    // a correctness check for the migration path.
    let kernel_name = std::env::var("ASCEND_SOFTMAX_KERNEL")
        .unwrap_or_else(|_| "tile_softmax".to_string());
    info!("using kernel: {}", kernel_name);

    // -- Configuration -------------------------------------------------------
    const ROWS: usize = 1;
    const COLS: usize = 1024;
    let n: usize = ROWS * COLS;

    info!("tile_softmax test: ROWS={}, COLS={}, n={}", ROWS, COLS, n);

    // -- Initialize ACL -------------------------------------------------------
    let acl = Acl::new()?;
    let device = Device::new(&acl)?;
    let context = AclContext::new(&device)?;
    let stream = AclStream::new(&context)?;
    info!("Device {} initialized", device.descriptor());

    // -- Generate synthetic input data ----------------------------------------
    let input_f32: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.01).sin() * 3.0)
        .collect();

    // -- CPU reference --------------------------------------------------------
    let expected = softmax_cpu(&input_f32, ROWS, COLS);

    // -- Run kernel -----------------------------------------------------------
    // Launch via the generated `<name>_do` extern "C" wrapper (the AscendC
    // launch entry that sets up the device launch context), dlsym'd from the
    // kernel object — the same path tile_matmul uses. Launching the raw stub
    // through rtKernelLaunch (KernelLoader::launch) instead fails ACL
    // "Parameter validation failed".
    info!("Launching tile_softmax kernel (1 block, {}x{} f32)...", ROWS, COLS);

    let mut d_input = DeviceBuffer::from_slice(&input_f32)?;
    let mut d_output = unsafe { DeviceBuffer::<f32>::uninitialized(n)? };

    // Args: (block_dim, stream, gm_input, gm_output)
    type KernelDoFn = unsafe extern "C" fn(u32, *mut c_void, *mut u8, *mut u8);
    let kernel_path = concat!(env!("OUT_DIR"), "/kernel.o");
    let kernel_lib =
        unsafe { libloading::Library::new(kernel_path) }.context("dlopen kernel.o")?;
    let do_name = format!("{}_do\0", kernel_name);
    unsafe {
        let kernel_fn: KernelDoFn = *kernel_lib
            .get(do_name.as_bytes())
            .with_context(|| format!("dlsym {}_do", kernel_name))?;
        kernel_fn(
            1, // block_dim — 1 block processes ROWS*COLS elements
            stream.to_raw(),
            d_input.as_mut_ptr() as *mut u8,
            d_output.as_mut_ptr() as *mut u8,
        );
    }
    stream.synchronize()?;

    // -- Verify results -------------------------------------------------------
    let output = d_output.to_host()?;

    let mut max_err = 0.0f32;
    for (npu, cpu) in output.iter().zip(expected.iter()) {
        let err = (npu - cpu).abs();
        if err > max_err {
            max_err = err;
        }
    }

    let npu_sum: f32 = output.iter().sum();
    let sum_ok = (npu_sum - 1.0).abs() < 1e-4;
    let err_ok = max_err < 1e-5;

    info!(
        "tile_softmax: max_err={:.4e} sum={:.6} sum_ok={} {}",
        max_err,
        npu_sum,
        sum_ok,
        if err_ok && sum_ok { "PASS" } else { "FAIL" }
    );

    if !err_ok {
        eprintln!("[FAIL] tile_softmax max_err={:.4e} exceeds 1e-5", max_err);
    }
    if !sum_ok {
        eprintln!("[FAIL] tile_softmax row sum={:.6} (expected ~1.0)", npu_sum);
    }

    if err_ok && sum_ok {
        info!("tile_softmax PASSED");
        Ok(())
    } else {
        std::process::exit(1);
    }
}
