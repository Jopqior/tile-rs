//! Provenance and run logging. Logs only; no attachments.

use std::process::Command;

use crate::cases::Case;
use crate::compare::{ATOL, RTOL};
use crate::gpu::describe_host;
use crate::status::{RunSummary, Status};

pub fn log_line(line: impl AsRef<str>) {
    println!("{}", line.as_ref());
}

pub fn provenance() {
    log_line(format!(
        "VERIFY_PROGRAM=metal-correctness version={}",
        env!("CARGO_PKG_VERSION")
    ));
    if let Ok(sha) = std::env::var("GITHUB_SHA") {
        log_line(format!("SOURCE_COMMIT={sha}"));
    } else if let Ok(out) = Command::new("git").args(["rev-parse", "HEAD"]).output() {
        if out.status.success() {
            log_line(format!(
                "SOURCE_COMMIT={}",
                String::from_utf8_lossy(&out.stdout).trim()
            ));
        } else {
            log_line("SOURCE_COMMIT=unknown");
        }
    } else {
        log_line("SOURCE_COMMIT=unknown");
    }
    let manifest = env!("CARGO_MANIFEST_DIR");
    let lock = std::path::Path::new(manifest).join("Cargo.lock");
    match std::fs::read(&lock) {
        Ok(bytes) => log_line(format!(
            "CARGO_LOCK_BYTES={} path={}",
            bytes.len(),
            lock.display()
        )),
        Err(e) => log_line(format!("CARGO_LOCK=missing ({e})")),
    }
    if let Ok(out) = Command::new("rustc").arg("-vV").output() {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            log_line(format!("RUSTC {line}"));
        }
    }
    if let Ok(out) = Command::new("cargo").arg("-V").output() {
        log_line(format!(
            "CARGO={}",
            String::from_utf8_lossy(&out.stdout).trim()
        ));
    }
    for line in describe_host() {
        log_line(line);
    }
    log_line("CACHE_POLICY=build cache allowed; emit, compile, reference, GPU execute always rerun");
    log_line(
        "COVERAGE=add/sub/mul/exp/add_mul elementwise family (issue #33); not the whole #31 first batch",
    );
}

/// Print small arrays in full, and summarize large generated arrays instead of
/// dumping thousands of values. The declared shape and generator config are
/// logged separately, and failure diagnostics still name the mismatch index.
fn log_array_summary(name: &str, values: &[f32]) {
    if values.len() <= 16 {
        log_line(format!("{name}={values:?}"));
    } else {
        let head = &values[..4];
        let tail = &values[values.len() - 4..];
        let zeros = values.iter().filter(|v| **v == 0.0).count();
        let negatives = values.iter().filter(|v| **v < 0.0).count();
        log_line(format!(
            "{name} len={} head={head:?} tail={tail:?} zeros={zeros} negatives={negatives}",
            values.len()
        ));
    }
}

pub fn log_case_config(case: &Case) {
    log_line(format!("CASE id={}", case.id));
    log_line(format!("CASE_MEANING={}", case.meaning));
    log_line(format!(
        "CASE_OP={} arity={}",
        case.op.tag(),
        case.op.arity()
    ));
    log_line(format!("CASE_SIZE={}", case.size_note));
    log_line(format!(
        "CASE_SHAPE={:?} dtype={} layout={}",
        case.shape, case.dtype, case.layout
    ));
    log_line(format!(
        "CASE_BINDINGS {} kernel={}",
        case.binding_note, case.kernel_name
    ));
    log_line(format!(
        "CASE_PRESERVE inputs=0..{} output_pad={} sentinel_init",
        case.op.arity() - 1,
        case.preserve_pad
    ));
    log_line(format!("CASE_COMPARE rtol={RTOL} atol={ATOL} finite_required=true"));
    log_line(format!(
        "CASE_ANCHOR={} generator={}",
        if case.hand_anchor { "literal_hand" } else { "independent_rust_op" },
        if case.hand_anchor { "literal".to_string() } else { format!("lcg64 seed={}", crate::cases::SEED) }
    ));
    for (i, x) in case.inputs.iter().enumerate() {
        log_array_summary(&format!("CASE_INPUT_{i}"), x);
    }
    log_array_summary("CASE_EXPECTED_ANCHOR", &case.expected_anchor);
}

pub fn log_msl_or_reason(msl: Option<&str>, reason: &str) {
    match msl {
        Some(src) if !src.is_empty() => {
            log_line("MSL_BEGIN");
            print!("{src}");
            if !src.ends_with('\n') {
                println!();
            }
            log_line("MSL_END");
        }
        _ => log_line(format!("MSL_NOT_GENERATED reason={reason}")),
    }
}

pub fn log_summary(summary: &RunSummary) {
    log_line(format!(
        "COMPLETE_LIST={}",
        if summary.complete_list { "yes" } else { "no" }
    ));
    log_line(format!(
        "CI_EQUIVALENT={}",
        if summary.complete_list { "yes" } else { "no" }
    ));
    log_line(format!("EXPECTED_LIST={}", summary.expected.join(",")));
    for r in &summary.results {
        log_line(format!(
            "CASE_STATUS id={} status={} stage={} detail={}",
            r.id,
            r.status.as_str(),
            r.stage,
            r.detail.replace('\n', " ")
        ));
        if let Some(skip) = &r.skipped_reason {
            log_line(format!("CASE_NOT_RUN id={} reason={skip}", r.id));
        }
    }
    for id in summary.missing() {
        log_line(format!("CASE_MISSING id={id}"));
    }
    let (pass, fail, unverified, missing) = summary.counts();
    log_line(format!(
        "RUN_SUMMARY pass={pass} fail={fail} unverified={unverified} missing={missing}"
    ));
    let overall = summary.overall();
    log_line(format!("RUN_RESULT={}", overall.as_str()));
    if overall != Status::Pass {
        log_line("RUN_SUCCESS=no");
    } else if !summary.complete_list {
        log_line("RUN_SUCCESS=single-case-only");
    } else {
        log_line("RUN_SUCCESS=yes");
    }
}

pub fn exit_code(summary: &RunSummary, self_check_ok: bool) -> i32 {
    if !self_check_ok {
        return 1;
    }
    // --case may exit 0 for local debug. The workflow never passes --case;
    // CI_EQUIVALENT=no is always logged in that mode.
    match summary.overall() {
        Status::Pass => 0,
        Status::Fail => 1,
        Status::Unverified => 2,
    }
}
