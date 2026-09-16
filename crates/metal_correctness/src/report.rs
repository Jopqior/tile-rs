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
    log_line("COVERAGE=initial add_f32_small only; not #31 first-batch complete");
}

pub fn log_case_config(case: &Case) {
    log_line(format!("CASE id={}", case.id));
    log_line(format!("CASE_MEANING={}", case.meaning));
    log_line(format!(
        "CASE_SHAPE={:?} dtype={} layout={}",
        case.shape, case.dtype, case.layout
    ));
    log_line(format!(
        "CASE_BINDINGS p0=a p1=b p2=out buffer(3)=num_elements kernel={}",
        case.kernel_name
    ));
    log_line(format!(
        "CASE_PRESERVE inputs=a,b output_pad={} sentinel_init",
        case.preserve_pad
    ));
    log_line(format!("CASE_COMPARE rtol={RTOL} atol={ATOL} finite_required=true"));
    log_line(format!("CASE_INPUT_A={:?}", case.a));
    log_line(format!("CASE_INPUT_B={:?}", case.b));
    log_line(format!("CASE_HAND_EXPECTED={:?}", case.hand_expected));
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
