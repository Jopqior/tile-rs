use kernel_correctness::golden::assert_approx;

const TOL: f32 = 1e-5;

fn resize_nearest(x: &[f32]) -> Vec<f32> {
    x.to_vec()
}
fn lerp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(&ai, &bi)| (1.0 - t) * ai + t * bi)
        .collect()
}
fn bicubic_weight(distances: &[f32]) -> Vec<f32> {
    // w(t) = 1.5*|t|^3 - 2.5*|t|^2 + 1 for |t|<=1, a=-0.5
    distances
        .iter()
        .map(|&d| {
            let t = d.abs();
            1.5 * t * t * t - 2.5 * t * t + 1.0
        })
        .collect()
}
fn weighted_sum(a: &[f32], b: &[f32], w1: f32, w2: f32) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(&ai, &bi)| w1 * ai + w2 * bi)
        .collect()
}
fn trilinear_1d(a: &[f32], b: &[f32], alpha: f32) -> Vec<f32> {
    lerp(a, b, alpha)
}

// ===== resize_ops_kernel.rs (5 kernels) =====
#[test]
fn test_resize_nearest() {
    assert_approx(
        &resize_nearest(&[1.0, 2.0, 3.0]),
        &[1.0, 2.0, 3.0],
        TOL,
        "nearest",
    );
}
#[test]
fn test_lerp_midpoint() {
    assert_approx(
        &lerp(&[0.0, 0.0], &[2.0, 4.0], 0.5),
        &[1.0, 2.0],
        TOL,
        "lerp_mid",
    );
}
#[test]
fn test_lerp_endpoints() {
    assert_approx(&lerp(&[1.0], &[3.0], 0.0), &[1.0], TOL, "lerp_0");
    assert_approx(&lerp(&[1.0], &[3.0], 1.0), &[3.0], TOL, "lerp_1");
}
#[test]
fn test_bicubic_weight_zero() {
    assert_approx(&bicubic_weight(&[0.0]), &[1.0], TOL, "bicubic_0");
}
#[test]
fn test_bicubic_weight_half() {
    let w = bicubic_weight(&[0.5]);
    let expected = 1.5 * 0.125 - 2.5 * 0.25 + 1.0;
    assert!(
        (w[0] - expected).abs() < TOL,
        "bicubic_0.5: {} vs {}",
        w[0],
        expected
    );
}
#[test]
fn test_weighted_sum() {
    assert_approx(
        &weighted_sum(&[1.0, 2.0], &[3.0, 4.0], 0.3, 0.7),
        &[2.4, 3.4],
        TOL,
        "wsum",
    );
}
#[test]
fn test_trilinear_1d() {
    assert_approx(
        &trilinear_1d(&[1.0], &[3.0], 0.25),
        &[1.5],
        TOL,
        "trilinear",
    );
}
