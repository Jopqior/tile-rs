//! CPU reference implementations of NPU kernel functions.
//!
//! These mirror the kernels in tests/compiletest/ui/*_kernel.rs
//! but use safe Rust with slices instead of raw pointers.
//! Used for golden-value correctness testing without NPU hardware.

pub mod activation;
pub mod broadcast;
pub mod conv;
pub mod fuse;
pub mod index;
pub mod kernel_family;
pub mod loss;
pub mod math;
pub mod matmul;
pub mod normalization;
pub mod optimizer;
pub mod pooling;
pub mod reduce;
pub mod resize;

/// Helper to load golden value JSON files.
pub mod golden {
    use serde::Deserialize;
    use std::path::Path;

    #[derive(Deserialize, Debug)]
    pub struct ConvCase {
        pub params: Vec<usize>,
        pub input: Vec<f32>,
        pub weight: Vec<f32>,
        pub expected: Vec<f32>,
    }

    #[derive(Deserialize, Debug)]
    pub struct IndexCase {
        pub input: Vec<f32>,
        pub index: Vec<usize>,
        pub expected: Vec<f32>,
        #[serde(default)]
        pub params: Vec<usize>,
    }

    #[derive(Deserialize, Debug)]
    pub struct PoolCase {
        pub params: Vec<usize>,
        pub input: Vec<f32>,
        pub expected: Vec<f32>,
    }

    #[derive(Deserialize, Debug)]
    pub struct MatmulCase {
        pub dims: Vec<usize>,
        pub a: Vec<f32>,
        pub b: Vec<f32>,
        pub expected: Vec<f32>,
    }

    #[derive(Deserialize, Debug)]
    pub struct ResizeCase {
        pub params: Vec<usize>,
        pub input: Vec<f32>,
        pub expected: Vec<f32>,
    }

    #[derive(Deserialize, Debug)]
    pub struct GenericCase {
        pub inputs: Vec<Vec<f32>>,
        pub expected: Vec<f32>,
        #[serde(default)]
        pub params_f: Vec<f32>,
        #[serde(default)]
        pub params_u: Vec<usize>,
    }

    pub fn load_json<T: serde::de::DeserializeOwned>(path: &str) -> Vec<T> {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let full = Path::new(&manifest).join(path);
        let data = std::fs::read_to_string(&full)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", full.display(), e));
        serde_json::from_str(&data).unwrap()
    }

    /// Assert two f32 slices are approximately equal.
    pub fn assert_approx(got: &[f32], want: &[f32], tol: f32, context: &str) {
        assert_eq!(
            got.len(),
            want.len(),
            "{context}: length mismatch: {} vs {}",
            got.len(),
            want.len()
        );
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let err = (g - w).abs();
            assert!(
                err < tol || (w.abs() > 1e-6 && err / w.abs() < tol),
                "{context} elem {i}: got {g}, want {w}, err {err}"
            );
        }
    }
}
