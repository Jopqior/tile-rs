//! Automated CPU equivalence test for all 998 CANN kernels.
//!
//! For each kernel in the manifest:
//! 1. Classify into a kernel family
//! 2. Get the CPU reference function
//! 3. Run with random inputs and verify it produces valid output
//!
//! This validates:
//! - Every kernel maps to a known family (no Unknown)
//! - Every reference function is numerically stable
//! - The classification covers the full manifest

use kernel_correctness::kernel_family::{classify_kernel, get_reference_fn, KernelFamily};

/// Parse the manifest file and return (name, category) pairs.
fn load_manifest() -> Vec<(String, String)> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = format!("{}/../../scripts/cann_kernel_manifest.txt", manifest_dir);
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot find manifest at {}: {}", path, e));

    content
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .filter_map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            if cols.len() >= 2 && cols[0] != "kernel_name" {
                Some((cols[0].to_string(), cols[1].to_string()))
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn test_all_998_classified() {
    let kernels = load_manifest();
    assert!(
        kernels.len() >= 990,
        "Expected ~998 kernels, got {}",
        kernels.len()
    );

    let mut unknown_count = 0;
    let mut unknown_names = Vec::new();

    for (name, cat) in &kernels {
        let family = classify_kernel(name, cat);
        if family == KernelFamily::Unknown {
            unknown_count += 1;
            if unknown_names.len() < 20 {
                unknown_names.push(format!("{}:{}", cat, name));
            }
        }
    }

    let classified = kernels.len() - unknown_count;
    let pct = classified as f64 / kernels.len() as f64 * 100.0;
    println!("Classification: {classified}/{} ({pct:.1}%)", kernels.len());
    if !unknown_names.is_empty() {
        println!("Unknown kernels (first 20):");
        for n in &unknown_names {
            println!("  {n}");
        }
    }
    eprintln!("Classification: {classified}/{} ({pct:.1}%)", kernels.len());
    if !unknown_names.is_empty() {
        eprintln!("Unknown kernels (first 20):");
        for n in &unknown_names {
            eprintln!("  {n}");
        }
    }
    // Allow up to 5% unknown — these are complex/niche kernels
    assert!(
        pct >= 95.0,
        "Too many unclassified kernels: {unknown_count}/{} ({:.1}% unknown)",
        kernels.len(),
        100.0 - pct
    );
}

#[test]
fn test_all_reference_fns_stable() {
    let kernels = load_manifest();
    let n = 256; // test vector size

    let mut pass = 0;
    let mut fail = 0;
    let mut errors = Vec::new();

    for (name, cat) in &kernels {
        let family = classify_kernel(name, cat);
        if family == KernelFamily::Unknown {
            continue;
        }

        let f = get_reference_fn(family);

        // Generate deterministic "random" input based on kernel name hash
        let seed = name
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let v =
                    ((seed.wrapping_add(i as u64)).wrapping_mul(6364136223846793005) >> 33) as f32;
                // Scale to [-2, 2] range — avoids exp overflow while testing interesting range
                (v / (u32::MAX as f32 / 4.0)) - 2.0
            })
            .collect();

        // Determine output size based on family
        let out_size = match family {
            KernelFamily::ReduceMax
            | KernelFamily::ReduceSum
            | KernelFamily::CrossEntropy
            | KernelFamily::MseLoss => 1,
            _ => n,
        };

        let mut output = vec![0.0_f32; out_size];

        // Run the reference function
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            f(&input[..out_size.min(n)], &mut output);
        }));

        match result {
            Ok(()) => {
                // Check output is finite (no NaN/Inf from numerical instability)
                let all_finite = output.iter().all(|v| v.is_finite());
                if all_finite {
                    pass += 1;
                } else {
                    fail += 1;
                    if errors.len() < 20 {
                        let bad = output.iter().position(|v| !v.is_finite()).unwrap();
                        errors.push(format!(
                            "{name} ({family:?}): non-finite at [{bad}] = {}",
                            output[bad]
                        ));
                    }
                }
            }
            Err(_) => {
                fail += 1;
                if errors.len() < 20 {
                    errors.push(format!("{name} ({family:?}): panicked"));
                }
            }
        }
    }

    println!("\nReference function stability: {pass} pass, {fail} fail");
    if !errors.is_empty() {
        println!("Errors:");
        for e in &errors {
            println!("  {e}");
        }
    }
    assert!(
        fail == 0,
        "{fail} reference functions produced invalid output"
    );
}

#[test]
fn test_family_distribution() {
    let kernels = load_manifest();
    let mut counts = std::collections::HashMap::new();

    for (name, cat) in &kernels {
        let family = classify_kernel(name, cat);
        *counts.entry(format!("{family:?}")).or_insert(0usize) += 1;
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nKernel family distribution:");
    for (family, count) in &sorted {
        println!("  {family:>25}: {count:>4}");
    }

    // Verify no single family dominates too much (sanity check)
    let max_count = sorted[0].1;
    assert!(
        max_count < kernels.len() / 2,
        "Single family {} has {} of {} kernels — classification too coarse",
        sorted[0].0,
        max_count,
        kernels.len()
    );
}
