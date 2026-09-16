//! Independent Metal correctness entry.
//!
//! MLIR → current-source `convert_mlir_to_msl` → Metal runtime compile →
//! GPU execute / readback → PyTorch CPU compare → status summary.
//!
//! This is staged coverage for issue #32 (one small f32 add case). Passing
//! the delivered list here is not #31 first-batch complete.

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

use cases::{case_by_id, delivered_cases, delivered_ids, required_inputs_present, Case};
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
Delivered list for this entry: add_f32_small. That is not #31 complete.
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
                let id = it.next().ok_or_else(|| "--case requires a case id".to_string())?;
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

fn run_one(case: &Case) -> (CaseResult, Option<String>) {
    log_case_config(case);
    if let Err(e) = required_inputs_present(case) {
        return (CaseResult::fail(case.id, "input", e), None);
    }

    let msl = match emit_msl(case.mlir) {
        Ok(s) => s,
        Err(e) => {
            log_msl_or_reason(None, &format!("emitter error: {e}"));
            return (
                CaseResult::fail(case.id, "emit", e),
                None,
            );
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
        return (
            CaseResult::fail(
                case.id,
                "emit",
                format!("generated MSL has no kernel void {}", case.kernel_name),
            ),
            Some(msl),
        );
    }

    let torch = match reference::add_f32_cpu(case) {
        Ok(t) => t,
        Err(RefError::Unverified(e)) => {
            log_line(format!("REFERENCE_UNVERIFIED {e}"));
            return (
                CaseResult::unverified(case.id, "reference", e),
                Some(msl),
            );
        }
        Err(RefError::Fail(e)) => {
            log_msl_or_reason(Some(&msl), "reference failed after emit");
            return (CaseResult::fail(case.id, "reference", e), Some(msl));
        }
    };
    log_line(format!("PYTORCH_VERSION={}", torch.version));
    log_line(format!(
        "REFERENCE_DEVICE=cpu dtype={} shape={:?}",
        torch.dtype, torch.shape
    ));
    if let Err(e) = reference::check_hand_anchor(case, &torch) {
        log_msl_or_reason(Some(&msl), "hand anchor vs PyTorch");
        return (
            CaseResult::fail(case.id, "hand_anchor", e.to_string()),
            Some(msl),
        );
    }
    log_line("HAND_ANCHOR=pass");

    let gpu = match gpu::run(case, &msl) {
        Ok(g) => g,
        Err(GpuError::Unverified(e)) => {
            log_line(format!("GPU_UNVERIFIED {e}"));
            return (
                CaseResult::unverified(case.id, "device", e),
                Some(msl),
            );
        }
        Err(GpuError::Fail(e)) => {
            log_msl_or_reason(Some(&msl), "GPU compile/execute failed");
            return (CaseResult::fail(case.id, "gpu", e), Some(msl));
        }
    };
    log_line(format!(
        "METAL_DEVICE={} low_power={} headless={} threadgroup={} groups={}",
        gpu.device_name, gpu.device_low_power, gpu.device_headless, gpu.threadgroup, gpu.groups
    ));

    if let Err(e) = require_dtype(&torch.dtype, case.dtype) {
        log_msl_or_reason(Some(&msl), "dtype");
        return (CaseResult::fail(case.id, "compare", e.to_string()), Some(msl));
    }
    let out_n = case.a.len();
    if gpu.out.len() < out_n {
        log_msl_or_reason(Some(&msl), "short readback");
        return (
            CaseResult::fail(
                case.id,
                "readback",
                format!("read {} values, need {out_n}", gpu.out.len()),
            ),
            Some(msl),
        );
    }
    if let Err(e) = require_shape(&torch.shape, case.shape) {
        log_msl_or_reason(Some(&msl), "reference shape");
        return (CaseResult::fail(case.id, "compare", e.to_string()), Some(msl));
    }
    if let Err(e) = require_shape(&[out_n], case.shape) {
        log_msl_or_reason(Some(&msl), "declared shape");
        return (CaseResult::fail(case.id, "compare", e.to_string()), Some(msl));
    }
    if let Err(e) = require_shape(&[gpu.out[..out_n].len()], &torch.shape) {
        log_msl_or_reason(Some(&msl), "readback shape vs reference");
        return (CaseResult::fail(case.id, "compare", e.to_string()), Some(msl));
    }
    log_line("READBACK_DTYPE=f32");
    if let Err(e) = assert_close(&gpu.out[..out_n], &torch.values) {
        log_msl_or_reason(Some(&msl), "numerical mismatch");
        return (CaseResult::fail(case.id, "compare", e.to_string()), Some(msl));
    }
    if let Err(e) = assert_preserved("input_a", &gpu.a, case.a) {
        log_msl_or_reason(Some(&msl), "input_a not preserved");
        return (CaseResult::fail(case.id, "preserve", e.to_string()), Some(msl));
    }
    if let Err(e) = assert_preserved("input_b", &gpu.b, case.b) {
        log_msl_or_reason(Some(&msl), "input_b not preserved");
        return (CaseResult::fail(case.id, "preserve", e.to_string()), Some(msl));
    }
    if case.preserve_pad > 0 {
        let pad = &gpu.out[out_n..];
        let expect_pad = vec![crate::cases::OUTPUT_SENTINEL; case.preserve_pad];
        if let Err(e) = assert_preserved("output_pad", pad, &expect_pad) {
            log_msl_or_reason(Some(&msl), "output pad not preserved");
            return (CaseResult::fail(case.id, "preserve", e.to_string()), Some(msl));
        }
    }
    log_line(format!("GPU_VERIFY=pass id={}", case.id));
    (CaseResult::pass(case.id), Some(msl))
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

    let complete_list = args.case.is_none();
    if !complete_list {
        log_line("COMPLETE_LIST=no");
        log_line("CI_EQUIVALENT=no");
    }

    let cases: Vec<Case> = if let Some(id) = &args.case {
        match case_by_id(id) {
            Some(c) => vec![c],
            None => {
                log_line(format!("CASE_UNKNOWN id={id}"));
                let summary = RunSummary {
                    results: vec![CaseResult::fail(id, "select", "unknown case id")],
                    expected: delivered_ids().into_iter().map(|s| s.to_string()).collect(),
                    complete_list: false,
                };
                log_summary(&summary);
                std::process::exit(1);
            }
        }
    } else {
        delivered_cases()
    };

    let expected: Vec<String> = if complete_list {
        delivered_ids().into_iter().map(|s| s.to_string()).collect()
    } else {
        cases.iter().map(|c| c.id.to_string()).collect()
    };

    let mut results = Vec::new();
    let mut halt: Option<String> = None;
    for case in cases.iter() {
        if let Some(reason) = &halt {
            let mut r = CaseResult::unverified(case.id, "not_run", reason.clone());
            r.skipped_reason = Some(reason.clone());
            results.push(r);
            continue;
        }
        let (r, _msl) = run_one(case);
        if !r.status.is_success() {
            halt = Some(format!(
                "stopped after {} status={} stage={}",
                r.id,
                r.status.as_str(),
                r.stage
            ));
        }
        results.push(r);
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
