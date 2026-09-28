use kernel_correctness::reduce::*;

const TOL: f32 = 1e-5;

// ===== reduce_ops_kernel + reduce_kernels =====
#[test]
fn test_reduce_max() {
    assert!((reduce_max(&[1.0, 5.0, 3.0, 2.0]) - 5.0).abs() < TOL);
}
#[test]
fn test_reduce_max_neg() {
    assert!((reduce_max(&[-3.0, -1.0, -5.0]) - (-1.0)).abs() < TOL);
}
#[test]
fn test_reduce_min() {
    assert!((reduce_min(&[1.0, 5.0, 3.0, 2.0]) - 1.0).abs() < TOL);
}
#[test]
fn test_reduce_min_neg() {
    assert!((reduce_min(&[-3.0, -1.0, -5.0]) - (-5.0)).abs() < TOL);
}
#[test]
fn test_reduce_sum() {
    assert!((reduce_sum(&[1.0, 2.0, 3.0, 4.0]) - 10.0).abs() < TOL);
}
#[test]
fn test_reduce_mean() {
    assert!((reduce_mean(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < TOL);
}
#[test]
fn test_reduce_prod() {
    assert!((reduce_prod(&[1.0, 2.0, 3.0, 4.0]) - 24.0).abs() < TOL);
}
#[test]
fn test_reduce_prod_one() {
    assert!((reduce_prod(&[1.0, 1.0, 1.0]) - 1.0).abs() < TOL);
}

// ===== math_ops_kernel: cumsum, cumprod =====
#[test]
fn test_cumsum() {
    kernel_correctness::golden::assert_approx(
        &cumsum(&[1.0, 2.0, 3.0, 4.0]),
        &[1.0, 3.0, 6.0, 10.0],
        TOL,
        "cumsum",
    );
}
#[test]
fn test_cumsum_exclusive() {
    kernel_correctness::golden::assert_approx(
        &cumsum_exclusive(&[1.0, 2.0, 3.0]),
        &[0.0, 1.0, 3.0],
        TOL,
        "cumsum_excl",
    );
}
#[test]
fn test_cumsum_reverse() {
    kernel_correctness::golden::assert_approx(
        &cumsum_reverse(&[1.0, 2.0, 3.0]),
        &[6.0, 5.0, 3.0],
        TOL,
        "cumsum_rev",
    );
}
#[test]
fn test_cumprod() {
    kernel_correctness::golden::assert_approx(
        &cumprod(&[1.0, 2.0, 3.0, 4.0]),
        &[1.0, 2.0, 6.0, 24.0],
        TOL,
        "cumprod",
    );
}
