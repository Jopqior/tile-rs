// =============================================================================
// tile_matmul: Rust tile-API matrix multiply on Ascend NPU
// =============================================================================
//
// Tests five tile_matmul_f32 monomorphizations (M×K×N shapes) for correctness
// and measures device execution time using AclEvent.
//
// Codegen path: Rust → MLIR → mlir_to_cpp (scalar triple-loop fallback).
// The PTO path (mlir_to_pto → pto.tmatmul → cube unit) is the future fast path.
//
// Usage:
//   tile_matmul [--csv <path>]

use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use std::env;
use std::ffi::c_void;
use std::fs;
use std::io::Write;

/// Kernel launch function signature matching the _do wrappers in the generated C++.
/// Args: (block_dim, stream, gm0, gm1, gm2)
type KernelDoFn = unsafe extern "C" fn(u32, *mut c_void, *mut u8, *mut u8, *mut u8);

/// (M, K, N, kernel_name)
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (16, 16, 16, "tile_matmul_16x16x16"),
    (32, 32, 32, "tile_matmul_32x32x32"),
    (64, 64, 64, "tile_matmul_64x64x64"),
    (16, 32, 16, "tile_matmul_16x32x16"),
    (32, 64, 32, "tile_matmul_32x64x32"),
];

const WARMUP_ITERS: usize = 1;
const BENCH_ITERS: usize = 10;

/// LCG PRNG for deterministic f32 data.
fn generate_matrix(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // scale to [-1, 1] to keep products in reasonable range
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// CPU reference: C[M×N] = A[M×K] @ B[K×N], row-major.
fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    c
}

/// Check NPU output against CPU reference, return (pass, max_err).
fn check_correctness(npu: &[f32], cpu: &[f32]) -> (bool, f32) {
    let max_err = npu
        .iter()
        .zip(cpu.iter())
        .map(|(&n, &c)| (n - c).abs())
        .fold(0.0f32, f32::max);
    // Scalar arithmetic accumulates differently from vector; allow 1e-2 for large tiles.
    (max_err < 1e-2, max_err)
}

struct BenchResult {
    times: Vec<f32>,
    pass: bool,
    max_err: f32,
}

fn bench_shape(
    m: usize,
    k: usize,
    n: usize,
    kernel_fn: KernelDoFn,
    stream: &AclStream,
) -> Result<BenchResult> {
    let a_host = generate_matrix(m * k, (m * k) as u64);
    let b_host = generate_matrix(k * n, (k * n) as u64 + 1);

    let mut a_dev = DeviceBuffer::from_slice_with_policy(
        a_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )
    .context("a_dev alloc")?;
    let mut b_dev = DeviceBuffer::from_slice_with_policy(
        b_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )
    .context("b_dev alloc")?;
    let mut c_dev = unsafe {
        DeviceBuffer::<f32>::uninitialized_with_policy(m * n, AclrtMemMallocPolicy::HugeFirst)
            .context("c_dev alloc")?
    };

    let block_dim: u32 = 1;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        unsafe {
            kernel_fn(
                block_dim,
                stream.to_raw(),
                a_dev.as_mut_ptr() as *mut u8,
                b_dev.as_mut_ptr() as *mut u8,
                c_dev.as_mut_ptr() as *mut u8,
            );
        }
        stream.synchronize()?;
    }

    // Timed iterations
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = AclEvent::new()?;
        let end = AclEvent::new()?;
        start.record(stream)?;
        unsafe {
            kernel_fn(
                block_dim,
                stream.to_raw(),
                a_dev.as_mut_ptr() as *mut u8,
                b_dev.as_mut_ptr() as *mut u8,
                c_dev.as_mut_ptr() as *mut u8,
            );
        }
        end.record(stream)?;
        stream.synchronize()?;
        times.push(AclEvent::elapsed_time(&start, &end)?);
    }

    // Correctness check
    let c_locked = c_dev.to_host().context("c_dev to_host")?;
    let npu_out = c_locked.as_slice();
    let cpu_ref = matmul_cpu(&a_host, &b_host, m, k, n);

    // Print first 4 elements for inspection
    for j in 0..4.min(m * n) {
        let (r, c) = (j / n, j % n);
        eprintln!(
            "  [check] [{r}][{c}] npu={:.6} cpu={:.6} err={:.2e}",
            npu_out[j],
            cpu_ref[j],
            (npu_out[j] - cpu_ref[j]).abs()
        );
    }

    let (pass, max_err) = check_correctness(npu_out, &cpu_ref);
    eprintln!("  [check] max_err={:.2e}", max_err);

    info!("  {}x{}x{}: times={:?}", m, k, n, times);
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

    // Load kernel via dlopen — the .so constructor registers the device binary.
    let kernel_path = concat!(env!("OUT_DIR"), "/kernel.o");
    let kernel_lib = unsafe { libloading::Library::new(kernel_path) }
        .context("dlopen kernel.o")?;

    let mut csv_lines: Vec<String> = Vec::new();
    let mut all_pass = true;

    for &(m, k, n, kernel_name) in SHAPES {
        let shape_label = format!("{}x{}x{}", m, k, n);
        info!("Benchmarking {} ({})", kernel_name, shape_label);

        let do_name = format!("{}_do\0", kernel_name);
        let kernel_fn: KernelDoFn = unsafe {
            *kernel_lib
                .get(do_name.as_bytes())
                .with_context(|| format!("dlsym {}", kernel_name))?
        };

        let result = bench_shape(m, k, n, kernel_fn, &stream)?;

        let status = if result.pass { "PASS" } else { "FAIL" };
        eprintln!(
            "[correctness] {} {}: {} (max_err={:.2e})",
            kernel_name, shape_label, status, result.max_err
        );
        if !result.pass {
            all_pass = false;
        }

        for (run, &t) in result.times.iter().enumerate() {
            let line = format!(
                "BENCH,{},{},{},{:.6}",
                shape_label, kernel_name, run, t
            );
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
