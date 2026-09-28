use crate::target::AscendTarget;

/// Severity of a validation diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Warning,
    Error,
}

/// A single validation finding.
#[derive(Clone, Debug)]
pub struct ValidationDiagnostic {
    pub severity: Severity,
    pub message: String,
    pub line: Option<usize>,
}

impl std::fmt::Display for ValidationDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let level = match self.severity {
            Severity::Warning => "warning",
            Severity::Error => "error",
        };
        if let Some(line) = self.line {
            write!(f, "{}: line {}: {}", level, line, self.message)
        } else {
            write!(f, "{}: {}", level, self.message)
        }
    }
}

/// Run all validation passes on the given C++ kernel source.
pub fn validate_kernel(source: &str, target: AscendTarget) -> Vec<ValidationDiagnostic> {
    let mut diags = Vec::new();
    check_entry_point(source, &mut diags);
    check_sync_barriers(source, target, &mut diags);
    check_buffer_sizes(source, target, &mut diags);
    diags
}

/// Check 1: Kernel must have an `__aicore__` entry point.
fn check_entry_point(source: &str, diags: &mut Vec<ValidationDiagnostic>) {
    if !source.contains("__aicore__") {
        diags.push(ValidationDiagnostic {
            severity: Severity::Error,
            message: "no __aicore__ entry point found".to_string(),
            line: None,
        });
    }
}

/// Check 2: If DMA patterns are present, sync barriers should be too.
///
/// On 310P targets (needs_explicit_barriers), missing barriers are errors.
/// On 910B targets, they are warnings (auto-sync may handle it).
fn check_sync_barriers(source: &str, target: AscendTarget, diags: &mut Vec<ValidationDiagnostic>) {
    let dma_patterns = [
        "DataCopy",
        "copy_gm_to_ubuf",
        "copy_ubuf_to_gm",
        "copy_gm_to_cbuf",
        "copy_cbuf_to_gm",
        "copy_ubuf_to_cbuf",
    ];
    let barrier_patterns = ["pipe_barrier", "PipeBarrier", "set_flag", "wait_flag"];

    let has_dma = dma_patterns.iter().any(|p| source.contains(p));
    let has_barrier = barrier_patterns.iter().any(|p| source.contains(p));

    if has_dma && !has_barrier {
        let severity = if target.needs_explicit_barriers() {
            Severity::Error
        } else {
            Severity::Warning
        };

        // Find first DMA line for context
        let line = find_first_match(source, &dma_patterns);

        diags.push(ValidationDiagnostic {
            severity,
            message: format!(
                "DMA operations found but no pipe_barrier/sync — \
                 required on {} (add pipe_barrier(PIPE_ALL) between DMA and compute)",
                target
            ),
            line,
        });
    }
}

/// Check 3: Scan `InitBuffer` / `pipe.InitBuffer` calls and validate buffer sizes
/// against the target's hardware limits.
fn check_buffer_sizes(source: &str, target: AscendTarget, diags: &mut Vec<ValidationDiagnostic>) {
    let ub_limit = target.ub_size();

    for (line_num, line) in source.lines().enumerate() {
        // Match patterns like: InitBuffer(buf, SIZE) or pipe.InitBuffer(buf, N, SIZE)
        // We look for numeric literals that could be buffer sizes.
        if !line.contains("InitBuffer") {
            continue;
        }

        // Extract the last numeric argument (the size in bytes/elements).
        // InitBuffer(queue, num_bufs, size) — size is last arg
        // pipe.InitBuffer(buf, size) — size is last arg
        if let Some(size) = extract_last_numeric_arg(line) {
            // InitBuffer sizes are in bytes
            if size > ub_limit {
                diags.push(ValidationDiagnostic {
                    severity: Severity::Error,
                    message: format!(
                        "InitBuffer size {} bytes exceeds {} UB limit of {} bytes",
                        size, target, ub_limit
                    ),
                    line: Some(line_num + 1),
                });
            }
        }
    }
}

/// Find the 1-based line number of the first occurrence of any pattern.
fn find_first_match(source: &str, patterns: &[&str]) -> Option<usize> {
    for (i, line) in source.lines().enumerate() {
        for pat in patterns {
            if line.contains(pat) {
                return Some(i + 1);
            }
        }
    }
    None
}

/// Extract the last numeric argument from a function call on a line.
/// E.g. `pipe.InitBuffer(inQueueX, 1, 32768)` → Some(32768)
fn extract_last_numeric_arg(line: &str) -> Option<usize> {
    // Find the last ')' and work backwards to find numeric tokens
    let trimmed = line.trim();
    let paren_end = trimmed.rfind(')')?;
    let paren_start = trimmed[..paren_end].rfind('(')?;
    let args_str = &trimmed[paren_start + 1..paren_end];

    // Take the last comma-separated token and try to parse it
    let last_arg = args_str.rsplit(',').next()?.trim();

    // Handle expressions like `256 * 1024`
    if last_arg.contains('*') {
        let parts: Option<usize> = last_arg
            .split('*')
            .map(|p| p.trim().parse::<usize>().ok())
            .try_fold(1usize, |acc, v| v.map(|n| acc * n));
        return parts;
    }

    last_arg.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_entry_point() {
        let diags = validate_kernel("void foo() {}", AscendTarget::Ascend910B3);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].severity, Severity::Error);
        assert!(diags[0].message.contains("__aicore__"));
    }

    #[test]
    fn valid_kernel_no_dma() {
        let src = r#"
            extern "C" __global__ __aicore__ void kernel_main() {
                // scalar only
            }
        "#;
        let diags = validate_kernel(src, AscendTarget::Ascend910B3);
        assert!(diags.is_empty());
    }

    #[test]
    fn dma_without_barrier_910b_warns() {
        let src = r#"
            extern "C" __global__ __aicore__ void kernel_main() {
                DataCopy(dst, src, count);
            }
        "#;
        let diags = validate_kernel(src, AscendTarget::Ascend910B3);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].severity, Severity::Warning);
    }

    #[test]
    fn dma_without_barrier_310p_errors() {
        let src = r#"
            extern "C" __global__ __aicore__ void kernel_main() {
                DataCopy(dst, src, count);
            }
        "#;
        let diags = validate_kernel(src, AscendTarget::Ascend310P1);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].severity, Severity::Error);
    }

    #[test]
    fn dma_with_barrier_ok() {
        let src = r#"
            extern "C" __global__ __aicore__ void kernel_main() {
                DataCopy(dst, src, count);
                pipe_barrier(PIPE_ALL);
            }
        "#;
        let diags = validate_kernel(src, AscendTarget::Ascend310P1);
        assert!(diags.is_empty());
    }

    #[test]
    fn buffer_size_over_limit() {
        let src = r#"
            extern "C" __global__ __aicore__ void kernel_main() {
                pipe.InitBuffer(inQueueX, 1, 300000);
            }
        "#;
        // 910B UB = 192KB = 196608 bytes → 300000 exceeds it
        let diags = validate_kernel(src, AscendTarget::Ascend910B3);
        assert!(
            diags
                .iter()
                .any(|d| d.severity == Severity::Error && d.message.contains("InitBuffer size"))
        );
    }

    #[test]
    fn buffer_size_within_limit() {
        let src = r#"
            extern "C" __global__ __aicore__ void kernel_main() {
                pipe.InitBuffer(inQueueX, 1, 32768);
            }
        "#;
        let diags = validate_kernel(src, AscendTarget::Ascend910B3);
        assert!(diags.is_empty());
    }

    #[test]
    fn buffer_size_multiply_expr() {
        let src = r#"
            extern "C" __global__ __aicore__ void kernel_main() {
                pipe.InitBuffer(inQueueX, 1, 256 * 1024);
            }
        "#;
        // 256 * 1024 = 262144 > 192KB (196608) on 910B
        let diags = validate_kernel(src, AscendTarget::Ascend910B3);
        assert!(diags.iter().any(|d| d.severity == Severity::Error));

        // 256KB = 262144 <= 256KB (262144) on 310P
        let diags = validate_kernel(src, AscendTarget::Ascend310P1);
        assert!(!diags.iter().any(|d| d.message.contains("InitBuffer size")));
    }

    #[test]
    fn extract_last_arg() {
        assert_eq!(
            extract_last_numeric_arg("pipe.InitBuffer(inQueueX, 1, 32768)"),
            Some(32768)
        );
        assert_eq!(
            extract_last_numeric_arg("InitBuffer(buf, 256 * 1024)"),
            Some(256 * 1024)
        );
        assert_eq!(extract_last_numeric_arg("no_parens"), None);
    }
}
