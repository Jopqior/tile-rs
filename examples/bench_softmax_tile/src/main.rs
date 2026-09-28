// =============================================================================
// Benchmark: Rust tile softmax kernels on Ascend NPU
// =============================================================================
//
// Measures device-side execution time using AclEvent for 6 tile-API softmax
// kernel variants across multiple ROWS×COLS shapes.
//
// Usage:
//   bench_softmax_tile [--csv <path>]

use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;
use std::env;
use std::ffi::c_void;
use std::fs;
use std::io::Write;

/// (rows, cols, kernel_name) — shapes to benchmark.
/// Only rows=1 shapes are correct: the mlir_to_cpp softmax decomposition uses
/// ReduceMax/ReduceSum over all rows*cols elements, which is only correct for
/// single-row tiles. Multi-row shapes (4x256, 16x256, 16x512) are included but
/// will fail correctness; they are kept for timing comparison only.
const SHAPES: &[(usize, usize, &str)] = &[
    (1, 1024, "tile_softmax_1x1024"),
    (1, 4096, "tile_softmax_1x4096"),
    (1, 8192, "tile_softmax_1x8192"),
    (4, 256, "tile_softmax_4x256"),
    (16, 256, "tile_softmax_16x256"),
    (16, 512, "tile_softmax_16x512"),
];

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

/// CPU reference row-wise softmax. Handles rows > 1 correctly.
fn softmax_cpu_rowwise(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (c, e) in exps.iter().enumerate() {
            out[r * cols + c] = e / sum;
        }
    }
    out
}

/// Check NPU output against CPU reference (row-wise), return (pass, max_err).
fn check_correctness(npu_output: &[f32], cpu_ref: &[f32], rows: usize, cols: usize) -> (bool, f32) {
    let mut max_err: f32 = 0.0;
    let mut all_sums_ok = true;

    for r in 0..rows {
        let npu_row = &npu_output[r * cols..(r + 1) * cols];
        let cpu_row = &cpu_ref[r * cols..(r + 1) * cols];
        let npu_sum: f32 = npu_row.iter().sum();
        if (npu_sum - 1.0).abs() > 0.01 {
            all_sums_ok = false;
            eprintln!(
                "  [check] row {} sum={:.6} (expected ~1.0)",
                r, npu_sum
            );
        }
        for (c, (&npu, &cpu)) in npu_row.iter().zip(cpu_row.iter()).enumerate() {
            let err = (npu - cpu).abs();
            if err > max_err {
                max_err = err;
            }
            if r == 0 && c < 4 {
                eprintln!(
                    "  [check] [{}][{}] npu={:.8} cpu={:.8} err={:.2e}",
                    r, c, npu, cpu, err
                );
            }
        }
    }
    eprintln!(
        "  [check] max_err={:.2e}, all_row_sums_ok={}",
        max_err, all_sums_ok
    );
    let pass = all_sums_ok && max_err < 1e-3;
    (pass, max_err)
}

struct BenchResult {
    times: Vec<f32>,
    pass: bool,
    max_err: f32,
}

fn bench_shape(
    rows: usize,
    cols: usize,
    kernel_name: &str,
    kernel_loader: &KernelLoader,
    stream: &AclStream,
) -> Result<BenchResult> {
    let n = rows * cols;
    let input_host = generate_input(n, n as u64);

    let mut input_device = DeviceBuffer::from_slice_with_policy(
        input_host.as_slice(),
        AclrtMemMallocPolicy::HugeFirst,
    )
    .context("input_device alloc")?;
    let mut output_device = unsafe {
        DeviceBuffer::<f32>::uninitialized_with_policy(n, AclrtMemMallocPolicy::HugeFirst)
            .context("output_device alloc")?
    };

    let kernel = kernel_loader
        .get_kernel(kernel_name)
        .with_context(|| format!("get_kernel({})", kernel_name))?;
    let block_dim: u32 = 1;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        unsafe {
            let mut args: [*mut c_void; 2] = [
                input_device.as_mut_ptr() as *mut _,
                output_device.as_mut_ptr() as *mut _,
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
            let mut args: [*mut c_void; 2] = [
                input_device.as_mut_ptr() as *mut _,
                output_device.as_mut_ptr() as *mut _,
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
    let output_locked = output_device
        .to_host()
        .context("to_host for correctness check")?;
    let npu_output = output_locked.as_slice();
    let cpu_ref = softmax_cpu_rowwise(&input_host, rows, cols);
    let (pass, max_err) = check_correctness(npu_output, &cpu_ref, rows, cols);

    info!("  {} {}x{}: times={:?}", kernel_name, rows, cols, times);
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
        KernelLoader::from_bin_path(kernel_path).context("KernelLoader::from_bin_path")?;

    let mut csv_lines: Vec<String> = Vec::new();
    let mut all_pass = true;

    for &(rows, cols, kernel_name) in SHAPES {
        let size_label = format!("{}x{}", rows, cols);
        info!("Benchmarking {} ({})", kernel_name, size_label);
        let result = bench_shape(rows, cols, kernel_name, &kernel_loader, &stream)?;

        let status = if result.pass { "PASS" } else { "FAIL" };
        eprintln!(
            "[correctness] {} {}: {} (max_err={:.2e})",
            kernel_name, size_label, status, result.max_err
        );
        if !result.pass {
            all_pass = false;
        }

        for (run, &t) in result.times.iter().enumerate() {
            let line = format!("BENCH,{},{},{},{:.6}", size_label, kernel_name, run, t);
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
