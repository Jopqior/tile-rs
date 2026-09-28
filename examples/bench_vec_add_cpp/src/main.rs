// =============================================================================
// Benchmark: C++ vectorized f16 vec_add kernel on Ascend NPU
// =============================================================================
//
// Measures device-side execution time using AclEvent for the C++ vec_add
// kernel across multiple input sizes. Outputs CSV-formatted results.
// Includes correctness verification against CPU reference.
//
// Usage:
//   bench_vec_add_cpp [--csv <path>]

use anyhow::Result;
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use std::env;
use std::ffi::c_void;
use std::fs;
use std::io::Write;

unsafe extern "C" {
    fn vec_add_bench_do(
        blockDim: u32,
        stream: *mut c_void,
        x: *mut u8,
        y: *mut u8,
        z: *mut u8,
        len_buf: *mut u8,
    );
}

const SIZES: &[usize] = &[256, 1024, 4096, 16384, 65536];
const MAX_BLOCKS: u32 = 8;
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

/// Choose block_dim based on total size. Use up to MAX_BLOCKS AICore blocks,
/// ensuring each block gets at least 256 elements (one tile).
fn choose_block_dim(total_size: usize) -> u32 {
    let min_per_block = 256usize;
    let max_blocks = (total_size / min_per_block).min(MAX_BLOCKS as usize).max(1);
    let mut block_dim = max_blocks as u32;
    while block_dim > 1 && total_size % (block_dim as usize) != 0 {
        block_dim -= 1;
    }
    block_dim
}

/// Simple LCG PRNG for deterministic f16 data generation.
fn generate_f16_input(n: usize, seed: u64) -> Vec<AclFloat16> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            AclFloat16::from(val)
        })
        .collect()
}

/// Check NPU output against CPU reference, return (pass, max_err).
fn check_correctness(npu_output: &[AclFloat16], x: &[AclFloat16], y: &[AclFloat16]) -> (bool, f32) {
    let mut max_err: f32 = 0.0;
    let mut mismatch_count = 0usize;
    for (i, &npu) in npu_output.iter().enumerate() {
        let expected = f32::from(x[i]) + f32::from(y[i]);
        let actual = f32::from(npu);
        let err = (actual - expected).abs();
        if err > max_err {
            max_err = err;
        }
        if i < 4 {
            eprintln!(
                "  [check] [{}] npu={:.6} expected={:.6} err={:.2e}",
                i, actual, expected, err
            );
        } else if err > 1e-2 && mismatch_count < 5 {
            eprintln!(
                "  [check] [{}] MISMATCH npu={:.6} expected={:.6} err={:.2e}",
                i, actual, expected, err
            );
            mismatch_count += 1;
        }
    }
    eprintln!("  [check] max_err={:.2e}", max_err);
    let pass = max_err < 1e-2;
    (pass, max_err)
}

struct BenchResult {
    times: Vec<f32>,
    pass: bool,
    max_err: f32,
}

fn bench_kernel(
    stream: &AclStream,
    size: usize,
) -> Result<BenchResult> {
    let block_dim = choose_block_dim(size);
    let per_block = size / (block_dim as usize);

    let x_host = generate_f16_input(size, size as u64);
    let y_host = generate_f16_input(size, (size as u64).wrapping_add(12345));
    let len_host: Vec<u32> = vec![per_block as u32];

    let mut x_device = DeviceBuffer::from_slice_with_policy(
        x_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )?;
    let mut y_device = DeviceBuffer::from_slice_with_policy(
        y_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )?;
    let mut len_device = DeviceBuffer::from_slice_with_policy(
        len_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )?;
    let mut z_device = unsafe {
        DeviceBuffer::<AclFloat16>::uninitialized_with_policy(size, AclrtMemMallocPolicy::HugeFirst)?
    };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        unsafe {
            vec_add_bench_do(
                block_dim,
                stream.to_raw(),
                x_device.as_mut_ptr() as *mut u8,
                y_device.as_mut_ptr() as *mut u8,
                z_device.as_mut_ptr() as *mut u8,
                len_device.as_mut_ptr() as *mut u8,
            );
        }
        stream.synchronize()?;
    }

    // Timed iterations
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start_event = AclEvent::new()?;
        let end_event = AclEvent::new()?;

        start_event.record(stream)?;

        unsafe {
            vec_add_bench_do(
                block_dim,
                stream.to_raw(),
                x_device.as_mut_ptr() as *mut u8,
                y_device.as_mut_ptr() as *mut u8,
                z_device.as_mut_ptr() as *mut u8,
                len_device.as_mut_ptr() as *mut u8,
            );
        }

        end_event.record(stream)?;
        stream.synchronize()?;

        let elapsed_ms = AclEvent::elapsed_time(&start_event, &end_event)?;
        times.push(elapsed_ms);
    }

    // Correctness check
    stream.synchronize()?;
    let output_locked = z_device.to_host()?;
    let npu_output = output_locked.as_slice();
    let (pass, max_err) = check_correctness(npu_output, &x_host, &y_host);

    info!(
        "  cpp_vector size={} block_dim={}: times={:?}",
        size, block_dim, times
    );
    Ok(BenchResult { times, pass, max_err })
}

fn main() -> Result<()> {
    SimpleLogger::new().init()?;

    let args: Vec<String> = env::args().collect();
    let csv_path = args
        .windows(2)
        .find(|w| w[0] == "--csv")
        .map(|w| w[1].clone());

    let acl = Acl::new()?;
    let device = Device::new(&acl)?;
    info!("Device initialized");
    let context = AclContext::new(&device)?;
    let stream = AclStream::new(&context)?;

    let mut csv_lines: Vec<String> = Vec::new();
    let mut all_pass = true;

    for &size in SIZES {
        info!("Benchmarking cpp_vector vec_add, size={}", size);
        let result = bench_kernel(&stream, size)?;

        let status = if result.pass { "PASS" } else { "FAIL" };
        eprintln!(
            "[correctness] cpp_vector size={}: {} (max_err={:.2e})",
            size, status, result.max_err
        );
        if !result.pass {
            all_pass = false;
        }

        for (run, &t) in result.times.iter().enumerate() {
            let line = format!("BENCH,{},cpp_vector,{},{:.6}", size, run, t);
            println!("{}", line);
            csv_lines.push(line);
        }
    }

    if let Some(path) = csv_path {
        let mut file = fs::File::create(&path)?;
        for line in &csv_lines {
            writeln!(file, "{}", line)?;
        }
        info!("CSV written to {}", path);
    }

    if !all_pass {
        eprintln!("[correctness] WARNING: Some correctness checks FAILED");
    }

    Ok(())
}
