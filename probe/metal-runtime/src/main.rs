//! Minimal Metal runtime probe for a free GitHub-hosted macOS runner.
//!
//! Answers issue #9 for the runtime path: is a Metal device visible, and can a
//! simple *custom* MSL compute kernel be compiled from source
//! (`Device::new_library_with_source`), executed, and read back with results
//! matching an independently computed CPU f32 reference?
//!
//! Honest reporting: every failure mode exits non-zero with a PROBE_RESULT=
//! line; only full bit-exact agreement prints PROBE_RESULT=pass.

use metal::*;

const N: usize = 16;
const SRC: &str = include_str!("vec_add.metal");

fn fill_shared_buffer(device: &Device, data: &[f32]) -> Buffer {
    let len = (data.len() * std::mem::size_of::<f32>()) as u64;
    let buf = device.new_buffer(len, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
    }
    buf
}

fn main() {
    println!("== Metal device enumeration ==");
    let devices = Device::all();
    println!("device_count={}", devices.len());
    for d in &devices {
        println!(
            "device name={:?} registry_id={} low_power={} headless={} max_threads_per_threadgroup={}",
            d.name(),
            d.registry_id(),
            d.is_low_power(),
            d.is_headless(),
            d.max_threads_per_threadgroup()
        );
    }

    let Some(device) = Device::system_default() else {
        eprintln!("PROBE_RESULT=device_missing: no system default Metal device");
        std::process::exit(2);
    };
    println!("system_default name={:?}", device.name());

    println!("== Runtime source compilation ==");
    let library = match device.new_library_with_source(SRC, &CompileOptions::new()) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("PROBE_RESULT=library_compile_failed: {e:?}");
            std::process::exit(3);
        }
    };
    let kernel = match library.get_function("vec_add", None) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("PROBE_RESULT=kernel_not_found: {e:?}");
            std::process::exit(4);
        }
    };
    let pipeline = match device.new_compute_pipeline_state(&kernel) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("PROBE_RESULT=pipeline_failed: {e:?}");
            std::process::exit(5);
        }
    };
    println!(
        "pipeline_max_total_threads_per_threadgroup={}",
        pipeline.max_total_threads_per_threadgroup()
    );

    println!("== Inputs and CPU reference ==");
    let a: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..N).map(|i| 1.0 - (i as f32) * 0.25).collect();
    // Independent CPU f32 reference: same declared computation, computed in Rust.
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    println!("input_a={a:?}");
    println!("input_b={b:?}");
    println!("cpu_expected={expected:?}");

    println!("== Dispatch and readback ==");
    let a_buf = fill_shared_buffer(&device, &a);
    let b_buf = fill_shared_buffer(&device, &b);
    let o_buf = device.new_buffer(
        (N * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&a_buf), 0);
    encoder.set_buffer(1, Some(&b_buf), 0);
    encoder.set_buffer(2, Some(&o_buf), 0);
    encoder.dispatch_threads(MTLSize::new(N as u64, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = o_buf.contents() as *const f32;
    let got: Vec<f32> = (0..N).map(|i| unsafe { *ptr.add(i) }).collect();
    println!("gpu_result={got:?}");

    println!("== Comparison ==");
    let mut bit_exact = 0usize;
    let mut max_abs_diff = 0.0f32;
    for i in 0..N {
        if got[i].to_bits() == expected[i].to_bits() {
            bit_exact += 1;
        }
        let d = (got[i] - expected[i]).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    println!("bit_exact={bit_exact}/{N} max_abs_diff={max_abs_diff}");

    if bit_exact == N {
        println!(
            "PROBE_RESULT=pass: device visible; runtime source compile ok; kernel executed; \
             all {N} outputs bit-exact vs independent CPU f32 reference"
        );
    } else {
        eprintln!(
            "PROBE_RESULT=mismatch: bit_exact={bit_exact}/{N} max_abs_diff={max_abs_diff}"
        );
        std::process::exit(6);
    }
}
