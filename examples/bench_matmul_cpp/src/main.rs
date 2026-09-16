// =============================================================================
// Benchmark: C++ f16->f32 matmul kernel on Ascend NPU (cube engine)
// =============================================================================
//
// Measures device-side execution time using AclEvent for the C++ matmul
// kernel (fixed 32x32x32). Outputs CSV-formatted results.
// Includes correctness verification against CPU reference.
//
// Usage:
//   bench_matmul_cpp [--csv <path>]

use anyhow::Result;
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use std::env;
use std::ffi::c_void;
use std::fs;
use std::io::Write;

unsafe extern "C" {
    fn matmul_bench_do(
        blockDim: u32,
        stream: *mut c_void,
        a: *mut u8,
        b: *mut u8,
        c: *mut u8,
    );
}

// C++ kernel is fixed at 32x32x32
const M: u32 = 32;
const K: u32 = 32;
const N: u32 = 32;
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

/// Simple LCG PRNG for deterministic f16 data generation.
fn generate_f16_data(n: usize, seed: u64) -> Vec<AclFloat16> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 0.5 - 0.25;
            AclFloat16::from(val)
        })
        .collect()
}

/// CPU reference matmul: C[m,n] = A[m,k] * B[k,n]
fn matmul_cpu(a: &[AclFloat16], b: &[AclFloat16], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += f32::from(a[i * k + p]) * f32::from(b[p * n + j]);
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Check NPU output against CPU reference, return (pass, max_err).
fn check_correctness(npu_output: &[f32], cpu_ref: &[f32]) -> (bool, f32) {
    let mut max_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    let mut mismatch_count = 0usize;
    for (i, (&npu, &cpu)) in npu_output.iter().zip(cpu_ref.iter()).enumerate() {
        let err = (npu - cpu).abs();
        let rel = if cpu.abs() > 1e-6 {
            err / cpu.abs()
        } else {
            err
        };
        if err > max_err {
            max_err = err;
        }
        if rel > max_rel_err {
            max_rel_err = rel;
        }
        if i < 4 {
            eprintln!(
                "  [check] [{}] npu={:.6} cpu={:.6} err={:.2e}",
                i, npu, cpu, err
            );
        } else if err > 0.1 && mismatch_count < 5 {
            eprintln!(
                "  [check] [{}] MISMATCH npu={:.6} cpu={:.6} err={:.2e}",
                i, npu, cpu, err
            );
            mismatch_count += 1;
        }
    }
    eprintln!(
        "  [check] max_abs_err={:.2e} max_rel_err={:.2e}",
        max_err, max_rel_err
    );
    let pass = max_rel_err < 0.1 || max_err < 0.05;
    (pass, max_err)
}

struct BenchResult {
    times: Vec<f32>,
    pass: bool,
    max_err: f32,
}

fn bench_matmul(stream: &AclStream) -> Result<BenchResult> {
    let m = M as usize;
    let k = K as usize;
    let n = N as usize;

    let a_host = generate_f16_data(m * k, (m as u64) * 1000 + (k as u64));
    let b_host = generate_f16_data(k * n, (k as u64) * 1000 + (n as u64));

    let mut a_device = DeviceBuffer::from_slice_with_policy(
        a_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )?;
    let mut b_device = DeviceBuffer::from_slice_with_policy(
        b_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )?;
    let mut c_device = unsafe {
        DeviceBuffer::<f32>::uninitialized_with_policy(m * n, AclrtMemMallocPolicy::HugeFirst)?
    };

    let block_dim: u32 = 1;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        unsafe {
            matmul_bench_do(
                block_dim,
                stream.to_raw(),
                a_device.as_mut_ptr() as *mut u8,
                b_device.as_mut_ptr() as *mut u8,
                c_device.as_mut_ptr() as *mut u8,
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
            matmul_bench_do(
                block_dim,
                stream.to_raw(),
                a_device.as_mut_ptr() as *mut u8,
                b_device.as_mut_ptr() as *mut u8,
                c_device.as_mut_ptr() as *mut u8,
            );
        }

        end_event.record(stream)?;
        stream.synchronize()?;

        let elapsed_ms = AclEvent::elapsed_time(&start_event, &end_event)?;
        times.push(elapsed_ms);
    }

    // Correctness check
    stream.synchronize()?;
    let output_locked = c_device.to_host()?;
    let npu_output = output_locked.as_slice();
    let cpu_ref = matmul_cpu(&a_host, &b_host, m, k, n);
    let (pass, max_err) = check_correctness(npu_output, &cpu_ref);

    info!("  cpp_matmul 32x32x32: times={:?}", times);
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

    let size_label = "32x32x32";
    info!("Benchmarking cpp_matmul {}", size_label);
    let result = bench_matmul(&stream)?;

    let status = if result.pass { "PASS" } else { "FAIL" };
    eprintln!(
        "[correctness] cpp_matmul {}: {} (max_err={:.2e})",
        size_label, status, result.max_err
    );

    for (run, &t) in result.times.iter().enumerate() {
        let line = format!("BENCH,{},cpp_matmul,{},{:.6}", size_label, run, t);
        println!("{}", line);
        csv_lines.push(line);
    }

    if let Some(path) = csv_path {
        let mut file = fs::File::create(&path)?;
        for line in &csv_lines {
            writeln!(file, "{}", line)?;
        }
        info!("CSV written to {}", path);
    }

    if !result.pass {
        eprintln!("[correctness] WARNING: Correctness check FAILED");
    }

    Ok(())
}
