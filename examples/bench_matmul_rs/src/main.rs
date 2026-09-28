// =============================================================================
// Benchmark: Rust f16->f32 matmul kernel on Ascend NPU (cube engine)
// =============================================================================
//
// Measures device-side execution time using AclEvent for the Rust matmul
// kernel across multiple matrix sizes. Outputs CSV results.
// Includes correctness verification against CPU reference.
//
// Usage:
//   bench_matmul_rs [--csv <path>]

use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use std::env;
use std::ffi::c_void;
use std::fs;
use std::io::Write;

// Matrix sizes: (m, k, n) — must be multiples of 16 for cube engine
const SIZES: &[(u32, u32, u32)] = &[
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
];
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

/// Simple LCG PRNG for deterministic f16 data generation.
fn generate_f16_data(n: usize, seed: u64) -> Vec<AclFloat16> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Keep values small to avoid f16 overflow in matmul
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 0.5 - 0.25;
            AclFloat16::from(val)
        })
        .collect()
}

/// CPU reference matmul: C[m,n] = A[m,k] * B[k,n] (f16 inputs, f32 output)
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
    // f16 matmul accumulates rounding errors; allow generous tolerance
    let pass = max_rel_err < 0.1 || max_err < 0.05;
    (pass, max_err)
}

struct BenchResult {
    times: Vec<f32>,
    pass: bool,
    max_err: f32,
}

fn bench_matmul(
    kernel_loader: &KernelLoader,
    stream: &AclStream,
    m: u32,
    k: u32,
    n: u32,
) -> Result<BenchResult> {
    let a_host = generate_f16_data((m * k) as usize, (m as u64) * 1000 + (k as u64));
    let b_host = generate_f16_data((k * n) as usize, (k as u64) * 1000 + (n as u64));
    let dims_host: Vec<u32> = vec![m, k, n];

    let mut a_device = DeviceBuffer::from_slice_with_policy(
        a_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )
    .context("a_device alloc")?;
    let mut b_device = DeviceBuffer::from_slice_with_policy(
        b_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )
    .context("b_device alloc")?;
    let mut dims_device = DeviceBuffer::from_slice_with_policy(
        dims_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )
    .context("dims_device alloc")?;
    let mut c_device = unsafe {
        DeviceBuffer::<f32>::uninitialized_with_policy(
            (m * n) as usize,
            AclrtMemMallocPolicy::HugeFirst,
        )
        .context("c_device alloc")?
    };

    let kernel = kernel_loader
        .get_kernel("matmul")
        .context("get_kernel(matmul)")?;
    let block_dim: u32 = 1;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        unsafe {
            let mut args: [*mut c_void; 4] = [
                a_device.as_mut_ptr() as *mut _,
                b_device.as_mut_ptr() as *mut _,
                c_device.as_mut_ptr() as *mut _,
                dims_device.as_mut_ptr() as *mut _,
            ];
            kernel.launch(block_dim, stream, &mut args)?;
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
            let mut args: [*mut c_void; 4] = [
                a_device.as_mut_ptr() as *mut _,
                b_device.as_mut_ptr() as *mut _,
                c_device.as_mut_ptr() as *mut _,
                dims_device.as_mut_ptr() as *mut _,
            ];
            kernel.launch(block_dim, stream, &mut args)?;
        }

        end_event.record(stream)?;
        stream.synchronize()?;

        let elapsed_ms = AclEvent::elapsed_time(&start_event, &end_event)?;
        times.push(elapsed_ms);
    }

    // Correctness check
    stream.synchronize()?;
    let output_locked = c_device
        .to_host()
        .context("to_host for correctness check")?;
    let npu_output = output_locked.as_slice();
    let cpu_ref = matmul_cpu(&a_host, &b_host, m as usize, k as usize, n as usize);
    let (pass, max_err) = check_correctness(npu_output, &cpu_ref);

    info!(
        "  rust_matmul {}x{}x{}: times={:?}",
        m, k, n, times
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

    let kernel_path = concat!(env!("OUT_DIR"), "/kernel.o");
    let kernel_loader =
        KernelLoader::from_bin_path_with_magic(kernel_path, KernelMagic::Cube)
            .context("KernelLoader::from_bin_path_with_magic(Cube)")?;

    let mut csv_lines: Vec<String> = Vec::new();
    let mut all_pass = true;

    for &(m, k, n) in SIZES {
        let size_label = format!("{}x{}x{}", m, k, n);
        info!("Benchmarking rust_matmul {}", size_label);
        let result = bench_matmul(&kernel_loader, &stream, m, k, n)?;

        let status = if result.pass { "PASS" } else { "FAIL" };
        eprintln!(
            "[correctness] rust_matmul {}: {} (max_err={:.2e})",
            size_label, status, result.max_err
        );
        if !result.pass {
            all_pass = false;
        }

        for (run, &t) in result.times.iter().enumerate() {
            let line = format!("BENCH,{},rust_matmul,{},{:.6}", size_label, run, t);
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
