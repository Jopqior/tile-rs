use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;

const MAX_ROWS: usize = 10; // Define MAX_ROWS as a constant

fn print_matrix<T: std::fmt::Display + Copy>(
    matrix: &[T],
    num_rows: i32,
    num_cols: i32,
) -> AclResult<()> {
    let rows = std::cmp::min(num_rows as usize, MAX_ROWS);
    let cols = num_cols as usize;
    for i in 0..rows {
        for j in 0..cols {
            let element = matrix[i * cols + j];
            print!("{:>10}", element); // Use trait bound for printing
        }
        println!();
    }

    if num_rows as usize > MAX_ROWS {
        println!("{:>10}", "......");
    }
    Ok(())
}

fn main() -> Result<()> {
    // Initialize logger with error handling
    SimpleLogger::new()
        .init()
        .context("Failed to initialize logger")?;
    info!("Logger initialized successfully.");

    let m = 16;
    let n = 16;
    let k = 16;
    let alpha = AclFloat16::from(2.0);
    let beta = AclFloat16::from(1.0);

    // Initialize ACL and log the result
    info!("Initializing ACL...");
    let mut acl = Acl::new()?;
    info!("ACL initialized successfully.");

    // Device ID configuration
    info!("Initializing device...");
    let device = Device::new(&acl)?;
    info!("device initialized successfully.");

    info!("Creating ACL context and stream...");
    let context = AclContext::new(&device)?;
    let stream = AclStream::new(&context)?;

    // 2. Set model directory and log the result
    acl.set_model_dir("test_data/op_models")
        .context("Failed to set model directory")?;
    info!("Model directory set to 'test_data/op_models'");

    // 3. Read input data from files
    info!("Attempting to read matrix A from 'test_data/data/matrix_a.bin'...");
    let host_matrix_a = common::read_buf_from_file::<AclFloat16>("test_data/data/matrix_a.bin");
    info!("Read {} elems for matrix A", host_matrix_a.len());

    info!("Attempting to read matrix B from 'test_data/data/matrix_b.bin'...");
    let host_matrix_b = common::read_buf_from_file::<AclFloat16>("test_data/data/matrix_b.bin");
    info!("Read {} elems for matrix B", host_matrix_b.len());

    info!("Attempting to read matrix C from 'test_data/data/matrix_c.bin'...");
    let mut host_matrix_c = common::read_buf_from_file::<AclFloat16>("test_data/data/matrix_c.bin");
    info!("Read {} elems for matrix C", host_matrix_c.len());

    // 4. Allocate device memory.
    info!("Copying matrix A from host to device...");
    let d_matrix_a = DeviceBuffer::from_slice(host_matrix_a.as_slice())?;
    info!("Copying matrix B from host to device...");
    let d_matrix_b = DeviceBuffer::from_slice(host_matrix_b.as_slice())?;
    info!("Copying matrix C from host to device...");
    let mut d_matrix_c = DeviceBuffer::from_slice(host_matrix_c.as_slice())?;
    info!("Copying `alpha` param from host to device...");
    let d_alpha = DeviceBox::new(&alpha)?;
    info!("Copying `beta` param from host to device...");
    let d_beta = DeviceBox::new(&beta)?;

    // 5. Run kernel.
    unsafe {
        ascend_rs_blas::acl_blas_gemm_ex(
            ascend_rs_blas::AclTransType::TransN,
            ascend_rs_blas::AclTransType::TransN,
            ascend_rs_blas::AclTransType::TransN,
            m,
            n,
            k,
            &d_alpha,
            &d_matrix_a,
            -1,
            &d_matrix_b,
            -1,
            &d_beta,
            &mut d_matrix_c,
            -1,
            ascend_rs_blas::AclComputeType::HighPrecision,
            &stream,
        )?;
    }

    // 4. Synchronize stream
    info!("Synchronizing the stream...");
    stream.synchronize()?;

    unsafe {
        let host_matrix_c_ptr = host_matrix_c.as_mut_ptr();
        let d_matrix_c_ptr = d_matrix_c.as_device_ptr();
        device_to_host(d_matrix_c_ptr, host_matrix_c_ptr, host_matrix_c.len())?;
    }

    // 5. Print results.
    info!("printing matrix a: ",);
    print_matrix(host_matrix_a.as_slice(), m, k)?;
    info!("printing matrix b: ",);
    print_matrix(host_matrix_b.as_slice(), k, n)?;
    info!("printing matrix c: ",);
    print_matrix(host_matrix_c.as_slice(), m, n)?;

    Ok(())
}
