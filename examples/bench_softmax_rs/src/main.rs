// =============================================================================
// Benchmark: Rust vectorized softmax kernel on Ascend NPU
// =============================================================================
//
// Measures device-side execution time using AclEvent for the Rust vector-
// intrinsic softmax kernel across multiple input sizes. Outputs CSV results.
// Includes correctness verification against CPU reference.
//
// Usage:
//   bench_softmax_rs [--csv <path>]

use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use std::env;
use std::ffi::c_void;
use std::fs;
use std::io::Write;

const SIZES: &[usize] = &[256, 1024, 4096, 16384];
const WARMUP_ITERS: usize = 1;
const BENCH_ITERS: usize = 10;

/// Simple LCG PRNG for deterministic f32 data generation.
fn generate_input(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// CPU reference softmax for correctness verification.
fn softmax_cpu(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Check NPU output against CPU reference, return (pass, max_err).
fn check_correctness(npu_output: &[f32], cpu_ref: &[f32]) -> (bool, f32) {
    let npu_sum: f32 = npu_output.iter().sum();
    let mut max_err: f32 = 0.0;
    let mut mismatch_count = 0usize;
    for (i, (&npu, &cpu)) in npu_output.iter().zip(cpu_ref.iter()).enumerate() {
        let err = (npu - cpu).abs();
        if err > max_err {
            max_err = err;
        }
        if i < 4 {
            eprintln!(
                "  [check] [{}] npu={:.8} cpu={:.8} err={:.2e}",
                i, npu, cpu, err
            );
        } else if err > 1e-4 && mismatch_count < 5 {
            eprintln!(
                "  [check] [{}] MISMATCH npu={:.8} cpu={:.8} err={:.2e}",
                i, npu, cpu, err
            );
            mismatch_count += 1;
        }
    }
    eprintln!(
        "  [check] sum(npu)={:.6} (expected ~1.0), max_err={:.2e}",
        npu_sum, max_err
    );
    let sum_ok = (npu_sum - 1.0).abs() < 0.01;
    let pass = sum_ok && max_err < 1e-3;
    (pass, max_err)
}

struct BenchResult {
    times: Vec<f32>,
    pass: bool,
    max_err: f32,
}

type KernelDoFn = unsafe extern "C" fn(u32, *mut c_void, *mut u8, *mut u8, *mut u8);

fn bench_size(
    kernel_fn: KernelDoFn,
    stream: &AclStream,
    size: usize,
) -> Result<BenchResult> {
    let input_host = generate_input(size, size as u64);
    let len_host: Vec<u32> = vec![size as u32];

    let mut input_device = DeviceBuffer::from_slice_with_policy(
        input_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )
    .context("input_device alloc")?;
    let mut len_device =
        DeviceBuffer::from_slice_with_policy(len_host.as_slice(), AclrtMemMallocPolicy::HugeFirst)
            .context("len_device alloc")?;
    let mut output_device = unsafe {
        DeviceBuffer::<f32>::uninitialized_with_policy(size, AclrtMemMallocPolicy::HugeFirst)
            .context("output_device alloc")?
    };

    let block_dim: u32 = 1;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        unsafe {
            kernel_fn(
                block_dim,
                stream.to_raw(),
                input_device.as_mut_ptr() as *mut u8,
                output_device.as_mut_ptr() as *mut u8,
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
            kernel_fn(
                block_dim,
                stream.to_raw(),
                input_device.as_mut_ptr() as *mut u8,
                output_device.as_mut_ptr() as *mut u8,
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
    let output_locked = output_device
        .to_host()
        .context("to_host for correctness check")?;
    let npu_output = output_locked.as_slice();
    let cpu_ref = softmax_cpu(&input_host);
    let (pass, max_err) = check_correctness(npu_output, &cpu_ref);

    info!(
        "  rust_vector size={}: times={:?}",
        size, times
    );
    Ok(BenchResult {
        times,
        pass,
        max_err,
    })
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

    // Load kernel .so via dlopen — the constructor registers the device binary.
    let kernel_path = concat!(env!("OUT_DIR"), "/kernel.o");
    let kernel_lib = unsafe { libloading::Library::new(kernel_path) }
        .context("dlopen kernel.o")?;

    // Load all 4 kernel variants
    let variants: Vec<(&str, libloading::Symbol<KernelDoFn>)> = vec![
        ("rust_kernel_ops", unsafe { kernel_lib.get(b"softmax_naive_do\0") }.context("softmax_naive_do")?),
        ("rust_manual_vec", unsafe { kernel_lib.get(b"softmax_do\0") }.context("softmax_do")?),
        ("rust_pipeline", unsafe { kernel_lib.get(b"softmax_pipeline_do\0") }.context("softmax_pipeline_do")?),
        ("rust_async", unsafe { kernel_lib.get(b"softmax_async_do\0") }.context("softmax_async_do")?),
    ];
    info!("Loaded {} kernel variants via dlopen", variants.len());

    let mut csv_lines: Vec<String> = Vec::new();
    let mut all_pass = true;

    for &size in SIZES {
        for (name, func) in &variants {
            info!("Benchmarking {} softmax, size={}", name, size);
            let result = bench_size(**func, &stream, size)?;

            let status = if result.pass { "PASS" } else { "FAIL" };
            eprintln!(
                "[correctness] {} size={}: {} (max_err={:.2e})",
                name, size, status, result.max_err
            );
            if !result.pass {
                all_pass = false;
            }

            for (run, &t) in result.times.iter().enumerate() {
                let line = format!("BENCH,{},{},{},{:.6}", size, name, run, t);
                println!("{}", line);
                csv_lines.push(line);
            }
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
