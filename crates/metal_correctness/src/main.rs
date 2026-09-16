//! Independent Metal correctness entry.
//!
//! MLIR → current-source `convert_mlir_to_msl` → Metal runtime compile →
//! GPU execute / readback → PyTorch CPU compare → status summary.
//!
//! This entry delivers the #33 elementwise family: `add`, `sub`, `mul`,
//! standalone `exp`, and the two-step combination `(a+b)*c`. Passing this
//! list is not the whole #31 first batch.

#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_parens,
    non_camel_case_types,
    unreachable_patterns,
    clippy::all
)]

// Emitter sources, included with no LLVM dep (same wiring as tile_spec).
#[path = "../../rustc_codegen_tile/src/mlir_parse.rs"]
mod mlir_parse;
#[path = "../../rustc_codegen_tile/src/mlir_to_pto.rs"]
mod mlir_to_pto;
#[path = "../../rustc_codegen_tile/src/mlir_to_msl.rs"]
mod mlir_to_msl;

mod cases;
mod compare;
mod gpu;
mod reference;
mod report;
mod self_check;
mod status;

use cases::{
    delivered_ids, delivered_specs, materialize, mlir_for, required_inputs_present, spec_by_id,
    Case, CaseSpec, DeviceCaps, RefOp,
};
use compare::{assert_close, assert_preserved, require_dtype, require_shape};
use gpu::GpuError;
use reference::RefError;
use report::{exit_code, log_case_config, log_line, log_msl_or_reason, log_summary, provenance};
use status::{CaseResult, RunSummary, Status};

struct Args {
    self_check_only: bool,
    list: bool,
    case: Option<String>,
}

fn print_help() {
    print!(
        "\
metal-correctness: current-source Metal emitter to GPU vs PyTorch CPU

Usage:
  metal-correctness                 run self-check then the full delivered list
  metal-correctness --self-check    comparator / completeness injections only
  metal-correctness --list          print the delivered case ids
  metal-correctness --case <id>     one case (debug; not CI-equivalent)

Re-run at a source version:
  git checkout <commit>
  cd crates/metal_correctness
  cargo run --release
  cargo run --release -- --case add_f32_small

CI must invoke the default (full list). --case cannot stand in for it.
Delivered list: add/sub/mul/exp/add_mul elementwise family (see --list).
That is not the whole #31 first batch.
"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        self_check_only: false,
        list: false,
        case: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--self-check" => args.self_check_only = true,
            "--list" => args.list = true,
            "--case" => {
                let id = it
                    .next()
                    .ok_or_else(|| "--case requires a case id".to_string())?;
                args.case = Some(id);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(args)
}

fn emit_msl(mlir: &str) -> Result<String, String> {
    mlir_to_msl::convert_mlir_to_msl(mlir)
}

/// Discover device capability through the current-source emitter, or explain
/// why it is unavailable. A probe emit/compile failure is a real failure, not
/// "unverified": the same emitter is under test.
fn discover_caps() -> Result<DeviceCaps, GpuError> {
    let probe_msl = mlir_for(RefOp::Add, "kc_caps_probe", 1);
    let msl = emit_msl(&probe_msl).map_err(|e| GpuError::Fail(format!("probe emit failed: {e}")))?;
    gpu::discover_caps(&msl, "kc_caps_probe")
}

/// Full per-case verification. Each stage reports its own failure so the
/// summary distinguishes emit, reference, device, compare, and preserve faults.
fn verify_one(case: &Case, caps: Option<&DeviceCaps>) -> CaseResult {
    log_case_config(case);
    if let Err(e) = required_inputs_present(case) {
        return CaseResult::fail(case.id.clone(), "input", e);
    }

    let msl = match emit_msl(&case.mlir) {
        Ok(s) => s,
        Err(e) => {
            log_msl_or_reason(None, &format!("emitter error: {e}"));
            return CaseResult::fail(case.id.clone(), "emit", e);
        }
    };
    log_line(format!(
        "EMIT=pass interface=convert_mlir_to_msl bytes={}",
        msl.len()
    ));
    if std::env::var_os("METAL_CORRECTNESS_DUMP_MSL").is_some() {
        log_msl_or_reason(Some(&msl), "dump");
    }
    if !msl.contains(&format!("kernel void {}", case.kernel_name)) {
        log_msl_or_reason(Some(&msl), "kernel name missing");
        return CaseResult::fail(
            case.id.clone(),
            "emit",
            format!("generated MSL has no kernel void {}", case.kernel_name),
        );
    }

    let torch = match reference::compute_cpu(case) {
        Ok(t) => t,
        Err(RefError::Unverified(e)) => {
            log_line(format!("REFERENCE_UNVERIFIED {e}"));
            return CaseResult::unverified(case.id.clone(), "reference", e);
        }
        Err(RefError::Fail(e)) => {
            log_msl_or_reason(Some(&msl), "reference failed after emit");
            return CaseResult::fail(case.id.clone(), "reference", e);
        }
    };
    log_line(format!("PYTORCH_VERSION={}", torch.version));
    log_line(format!(
        "REFERENCE_DEVICE=cpu dtype={} shape={:?}",
        torch.dtype, torch.shape
    ));
    if let Err(e) = reference::check_anchor(case, &torch) {
        log_msl_or_reason(Some(&msl), "anchor vs PyTorch");
        return CaseResult::fail(case.id.clone(), "anchor", e.to_string());
    }
    log_line(if case.hand_anchor {
        "HAND_ANCHOR=pass"
    } else {
        "DEFINITION_ANCHOR=pass"
    });

    let Some(caps) = caps else {
        log_line("GPU_UNVERIFIED no Metal device caps available");
        return CaseResult::unverified(
            case.id.clone(),
            "device",
            "no Metal device caps available",
        );
    };
    let gpu = match gpu::run(case, &msl, caps) {
        Ok(g) => g,
        Err(GpuError::Unverified(e)) => {
            log_line(format!("GPU_UNVERIFIED {e}"));
            return CaseResult::unverified(case.id.clone(), "device", e);
        }
        Err(GpuError::Fail(e)) => {
            log_msl_or_reason(Some(&msl), "GPU compile/execute failed");
            return CaseResult::fail(case.id.clone(), "gpu", e);
        }
    };
    log_line(format!(
        "METAL_DEVICE={} low_power={} headless={} threadgroup={} groups={} simd_width={}",
        gpu.device_name,
        gpu.device_low_power,
        gpu.device_headless,
        gpu.threadgroup,
        gpu.groups,
        gpu.thread_execution_width
    ));

    let n = case.output_len();
    if let Err(e) = require_dtype(&torch.dtype, case.dtype) {
        log_msl_or_reason(Some(&msl), "dtype");
        return CaseResult::fail(case.id.clone(), "compare", e.to_string());
    }
    if gpu.out.len() < n {
        log_msl_or_reason(Some(&msl), "short readback");
        return CaseResult::fail(
            case.id.clone(),
            "readback",
            format!("read {} values, need {n}", gpu.out.len()),
        );
    }
    if let Err(e) = require_shape(&torch.shape, &case.shape) {
        log_msl_or_reason(Some(&msl), "reference shape");
        return CaseResult::fail(case.id.clone(), "compare", e.to_string());
    }
    if let Err(e) = require_shape(&[n], &case.shape) {
        log_msl_or_reason(Some(&msl), "declared shape");
        return CaseResult::fail(case.id.clone(), "compare", e.to_string());
    }
    if let Err(e) = require_shape(&[gpu.out[..n].len()], &torch.shape) {
        log_msl_or_reason(Some(&msl), "readback shape vs reference");
        return CaseResult::fail(case.id.clone(), "compare", e.to_string());
    }
    log_line("READBACK_DTYPE=f32");
    if let Err(e) = assert_close(&gpu.out[..n], &torch.values) {
        log_msl_or_reason(Some(&msl), "numerical mismatch");
        return CaseResult::fail(case.id.clone(), "compare", e.to_string());
    }
    if gpu.inputs.len() != case.inputs.len() {
        log_msl_or_reason(Some(&msl), "input readback count");
        return CaseResult::fail(
            case.id.clone(),
            "readback",
            format!(
                "read back {} input buffers, expected {}",
                gpu.inputs.len(),
                case.inputs.len()
            ),
        );
    }
    for (i, (actual, expected)) in gpu.inputs.iter().zip(case.inputs.iter()).enumerate() {
        if let Err(e) = assert_preserved(&format!("input_{i}"), actual, expected) {
            log_msl_or_reason(Some(&msl), "input not preserved");
            return CaseResult::fail(case.id.clone(), "preserve", e.to_string());
        }
    }
    if case.preserve_pad > 0 {
        let pad = &gpu.out[n..];
        let expect_pad = vec![crate::cases::OUTPUT_SENTINEL; case.preserve_pad];
        if let Err(e) = assert_preserved("output_pad", pad, &expect_pad) {
            log_msl_or_reason(Some(&msl), "output pad not preserved");
            return CaseResult::fail(case.id.clone(), "preserve", e.to_string());
        }
    }
    log_line(format!("GPU_VERIFY=pass id={}", case.id));
    CaseResult::pass(case.id.clone())
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("try --help");
            std::process::exit(1);
        }
    };

    if args.list {
        for id in delivered_ids() {
            println!("{id}");
        }
        log_line(format!("EXPECTED_LIST={}", delivered_ids().join(",")));
        std::process::exit(0);
    }

    provenance();

    let check = self_check::run();
    for line in &check.lines {
        log_line(line);
    }
    if !check.ok {
        log_line("RUN_RESULT=fail");
        log_line("RUN_SUCCESS=no");
        std::process::exit(1);
    }
    if args.self_check_only {
        log_line("GPU_VERIFY=not_run reason=self-check-only");
        log_line("RUN_RESULT=pass");
        std::process::exit(0);
    }

    // Only the cases that need device-relative sizes force capability
    // discovery; --case on a fixed case still works off-device.
    let needs_caps = match &args.case {
        Some(id) => spec_by_id(id).map(|s| s.size.needs_device()).unwrap_or(false),
        None => true,
    };
    let caps = if needs_caps {
        match discover_caps() {
            Ok(c) => Some(c),
            Err(GpuError::Unverified(e)) => {
                log_line(format!("DEVICE_CAPS_UNVERIFIED {e}"));
                None
            }
            Err(GpuError::Fail(e)) => {
                log_line(format!("DEVICE_CAPS_FAIL {e}"));
                log_line("RUN_RESULT=fail");
                log_line("RUN_SUCCESS=no");
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    let complete_list = args.case.is_none();
    if !complete_list {
        log_line("COMPLETE_LIST=no");
        log_line("CI_EQUIVALENT=no");
    }

    let specs: Vec<CaseSpec> = if let Some(id) = &args.case {
        match spec_by_id(id) {
            Some(s) => vec![s],
            None => {
                log_line(format!("CASE_UNKNOWN id={id}"));
                let summary = RunSummary {
                    results: vec![CaseResult::fail(id.clone(), "select", "unknown case id")],
                    expected: delivered_ids(),
                    complete_list: false,
                };
                log_summary(&summary);
                std::process::exit(1);
            }
        }
    } else {
        delivered_specs()
    };

    let expected: Vec<String> = if complete_list {
        delivered_ids()
    } else {
        specs.iter().map(|s| s.id.clone()).collect()
    };

    let mut results = Vec::new();
    for spec in specs.iter() {
        match materialize(spec, caps.as_ref()) {
            Ok(case) => {
                let r = verify_one(&case, caps.as_ref());
                results.push(r);
            }
            Err(e) => {
                log_line(format!("CASE_UNAVAILABLE id={} reason={e}", spec.id));
                let mut r = CaseResult::unverified(spec.id.clone(), "device", e);
                r.skipped_reason = Some("no device to instantiate case size".into());
                results.push(r);
            }
        }
    }

    let summary = RunSummary {
        results,
        expected,
        complete_list,
    };
    log_summary(&summary);
    let code = exit_code(&summary, check.ok);
    std::process::exit(code);
}
