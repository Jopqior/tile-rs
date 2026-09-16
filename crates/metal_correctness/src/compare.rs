//! f32 comparison policy from issue #31 / #11.
//!
//! |actual - expected| <= atol + rtol * |expected|
//! with rtol=1.3e-6, atol=1e-5. NaN/Inf on either side is a failure.
//! Preserved regions are bit-identical, not tolerant.

pub const RTOL: f32 = 1.3e-6;
pub const ATOL: f32 = 1e-5;

#[derive(Debug)]
pub struct Mismatch {
    pub index: usize,
    pub actual: f32,
    pub expected: f32,
    pub abs_diff: f32,
    pub tol: f32,
}

#[derive(Debug)]
pub enum CompareError {
    Shape {
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    Dtype {
        actual: String,
        expected: String,
    },
    NonFinite {
        side: &'static str,
        index: usize,
        value: f32,
    },
    Value(Mismatch),
    Preserve {
        region: String,
        index: usize,
        actual: f32,
        expected: f32,
    },
}

impl std::fmt::Display for CompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompareError::Shape { actual, expected } => {
                write!(f, "shape mismatch actual={actual:?} expected={expected:?}")
            }
            CompareError::Dtype { actual, expected } => {
                write!(f, "dtype mismatch actual={actual} expected={expected}")
            }
            CompareError::NonFinite { side, index, value } => {
                write!(f, "non-finite {side}[{index}]={value}")
            }
            CompareError::Value(m) => write!(
                f,
                "mismatch index={} actual={} expected={} abs_diff={} tol={}",
                m.index, m.actual, m.expected, m.abs_diff, m.tol
            ),
            CompareError::Preserve {
                region,
                index,
                actual,
                expected,
            } => write!(
                f,
                "preserve {region}[{index}] actual={actual} expected={expected}"
            ),
        }
    }
}

pub fn require_dtype(actual: &str, expected: &str) -> Result<(), CompareError> {
    if actual == expected {
        Ok(())
    } else {
        Err(CompareError::Dtype {
            actual: actual.to_string(),
            expected: expected.to_string(),
        })
    }
}

pub fn require_shape(actual: &[usize], expected: &[usize]) -> Result<(), CompareError> {
    if actual == expected {
        Ok(())
    } else {
        Err(CompareError::Shape {
            actual: actual.to_vec(),
            expected: expected.to_vec(),
        })
    }
}

pub fn assert_close(actual: &[f32], expected: &[f32]) -> Result<(), CompareError> {
    require_shape(&[actual.len()], &[expected.len()])?;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if !a.is_finite() {
            return Err(CompareError::NonFinite {
                side: "actual",
                index: i,
                value: a,
            });
        }
        if !e.is_finite() {
            return Err(CompareError::NonFinite {
                side: "expected",
                index: i,
                value: e,
            });
        }
        let abs_diff = (a - e).abs();
        let tol = ATOL + RTOL * e.abs();
        if abs_diff > tol {
            return Err(CompareError::Value(Mismatch {
                index: i,
                actual: a,
                expected: e,
                abs_diff,
                tol,
            }));
        }
    }
    Ok(())
}

/// Bit-identical preservation. Used for inputs and padding that the kernel
/// must not write.
pub fn assert_preserved(
    region: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), CompareError> {
    require_shape(&[actual.len()], &[expected.len()])?;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.to_bits() != e.to_bits() {
            return Err(CompareError::Preserve {
                region: region.to_string(),
                index: i,
                actual: a,
                expected: e,
            });
        }
    }
    Ok(())
}

/// True when `assert_close` would reject this pair. Used by self-check to
/// confirm the comparator is not a no-op.
pub fn would_reject(actual: &[f32], expected: &[f32]) -> bool {
    assert_close(actual, expected).is_err()
}
