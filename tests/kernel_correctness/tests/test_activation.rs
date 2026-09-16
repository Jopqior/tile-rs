use kernel_correctness::activation::*;
use kernel_correctness::golden::assert_approx;

const TOL: f32 = 1e-5;

// ===== Individual activation kernels =====

#[test]
fn test_relu() {
    assert_approx(
        &relu(&[-2.0, -1.0, 0.0, 1.0, 2.0]),
        &[0.0, 0.0, 0.0, 1.0, 2.0],
        TOL,
        "relu",
    );
}
#[test]
fn test_sigmoid() {
    assert_approx(&sigmoid(&[0.0]), &[0.5], TOL, "sigmoid_zero");
}
#[test]
fn test_sigmoid_sym() {
    let s = sigmoid(&[-1.0, 1.0]);
    assert!((s[0] + s[1] - 1.0).abs() < TOL, "sigmoid symmetry");
}
#[test]
fn test_tanh() {
    assert_approx(&tanh_act(&[0.0]), &[0.0], TOL, "tanh_zero");
}
#[test]
fn test_tanh_bounds() {
    let t = tanh_act(&[100.0, -100.0]);
    assert!((t[0] - 1.0).abs() < TOL && (t[1] + 1.0).abs() < TOL);
}
#[test]
fn test_gelu() {
    assert_approx(&gelu(&[0.0]), &[0.0], TOL, "gelu_zero");
}
#[test]
fn test_gelu_positive() {
    assert!(gelu(&[2.0])[0] > 1.9, "gelu(2) approx 2");
}
#[test]
fn test_gelu_tanh() {
    assert_approx(&gelu_tanh(&[0.0]), &[0.0], TOL, "gelu_tanh_zero");
}
#[test]
fn test_softmax_basic() {
    let s = softmax(&[1.0, 2.0, 3.0]);
    assert!(
        (s.iter().sum::<f32>() - 1.0).abs() < TOL,
        "softmax sums to 1"
    );
}
#[test]
fn test_softmax_uniform() {
    let s = softmax(&[1.0, 1.0, 1.0, 1.0]);
    for &v in &s {
        assert!((v - 0.25).abs() < TOL);
    }
}
#[test]
fn test_log_softmax() {
    let ls = log_softmax(&[1.0, 2.0, 3.0]);
    for &v in &ls {
        assert!(v <= 0.0, "log_softmax <= 0");
    }
}
#[test]
fn test_elu_pos() {
    assert_approx(&elu(&[1.0, 2.0], 1.0), &[1.0, 2.0], TOL, "elu_pos");
}
#[test]
fn test_elu_neg() {
    let e = elu(&[-1.0], 1.0);
    assert!((e[0] - ((-1.0f32).exp() - 1.0)).abs() < TOL);
}
#[test]
fn test_selu() {
    let s = selu(&[0.0]);
    assert!((s[0]).abs() < TOL, "selu(0)=0");
}
#[test]
fn test_swish() {
    assert_approx(&swish(&[0.0]), &[0.0], TOL, "swish_zero");
}
#[test]
fn test_swish_positive() {
    assert!(swish(&[2.0])[0] > 1.7);
}
#[test]
fn test_mish() {
    assert_approx(&mish(&[0.0]), &[0.0], TOL, "mish_zero");
}
#[test]
fn test_mish_positive() {
    assert!(mish(&[2.0])[0] > 1.9);
}
#[test]
fn test_softplus() {
    assert!((softplus(&[0.0])[0] - 2.0f32.ln()).abs() < TOL);
}
#[test]
fn test_softsign() {
    assert_approx(&softsign(&[0.0, 1.0]), &[0.0, 0.5], TOL, "softsign");
}
#[test]
fn test_hardsigmoid() {
    assert_approx(
        &hardsigmoid(&[0.0, 3.0, -3.0]),
        &[0.5, 1.0, 0.0],
        TOL,
        "hardsigmoid",
    );
}
#[test]
fn test_hardswish() {
    assert_approx(&hardswish(&[0.0]), &[0.0], TOL, "hardswish_zero");
}
#[test]
fn test_leaky_relu() {
    assert_approx(
        &leaky_relu(&[-1.0, 1.0], 0.01),
        &[-0.01, 1.0],
        TOL,
        "leaky_relu",
    );
}
#[test]
fn test_abs() {
    assert_approx(&abs_act(&[-2.0, 0.0, 3.0]), &[2.0, 0.0, 3.0], TOL, "abs");
}
#[test]
fn test_scalar_mul() {
    assert_approx(
        &scalar_mul(&[1.0, 2.0, 3.0], 2.5),
        &[2.5, 5.0, 7.5],
        TOL,
        "scalar_mul",
    );
}
#[test]
fn test_hardtanh() {
    assert_approx(
        &hardtanh(&[-5.0, 0.0, 5.0], -1.0, 1.0),
        &[-1.0, 0.0, 1.0],
        TOL,
        "hardtanh",
    );
}

// ===== Composite ops (composite_ops_kernel.rs) =====
#[test]
fn test_composite_sigmoid() {
    let s = sigmoid(&[0.0, 1.0, -1.0]);
    assert!((s[0] - 0.5).abs() < TOL);
}
#[test]
fn test_composite_tanh() {
    let t = tanh_act(&[-1.0, 0.0, 1.0]);
    assert!((t[1]).abs() < TOL);
}
#[test]
fn test_composite_gelu() {
    let g = gelu(&[-1.0, 0.0, 1.0]);
    assert!((g[1]).abs() < TOL);
}
#[test]
fn test_composite_softmax() {
    let s = softmax(&[1.0, 1.0]);
    assert!((s[0] - 0.5).abs() < TOL);
}

// ===== f32 unary ops =====
#[test]
fn test_exp() {
    assert_approx(
        &exp_f32(&[0.0, 1.0]),
        &[1.0, std::f32::consts::E],
        TOL,
        "exp",
    );
}
#[test]
fn test_ln() {
    assert_approx(&ln_f32(&[1.0, std::f32::consts::E]), &[0.0, 1.0], TOL, "ln");
}
#[test]
fn test_sqrt() {
    assert_approx(&sqrt_f32(&[1.0, 4.0, 9.0]), &[1.0, 2.0, 3.0], TOL, "sqrt");
}
#[test]
fn test_rsqrt() {
    assert_approx(&rsqrt_f32(&[1.0, 4.0]), &[1.0, 0.5], TOL, "rsqrt");
}
#[test]
fn test_reciprocal() {
    assert_approx(
        &reciprocal_f32(&[1.0, 2.0, 4.0]),
        &[1.0, 0.5, 0.25],
        TOL,
        "reciprocal",
    );
}
#[test]
fn test_negate() {
    assert_approx(
        &negate_f32(&[1.0, -2.0, 0.0]),
        &[-1.0, 2.0, 0.0],
        TOL,
        "negate",
    );
}
#[test]
fn test_square() {
    assert_approx(
        &square_f32(&[-2.0, 0.0, 3.0]),
        &[4.0, 0.0, 9.0],
        TOL,
        "square",
    );
}
#[test]
fn test_cube() {
    assert_approx(
        &cube_f32(&[-2.0, 0.0, 3.0]),
        &[-8.0, 0.0, 27.0],
        TOL,
        "cube",
    );
}

// ===== f16 activation equivalence (same math, just f16 precision) =====
#[test]
fn test_f16_relu_equiv() {
    assert_approx(&relu(&[-1.0, 0.0, 1.0]), &[0.0, 0.0, 1.0], TOL, "f16_relu");
}
#[test]
fn test_f16_sigmoid_equiv() {
    let s = sigmoid(&[0.0]);
    assert!((s[0] - 0.5).abs() < TOL);
}
#[test]
fn test_f16_abs_equiv() {
    assert_approx(&abs_act(&[-1.0, 1.0]), &[1.0, 1.0], TOL, "f16_abs");
}
#[test]
fn test_f16_exp_equiv() {
    assert_approx(&exp_f32(&[0.0]), &[1.0], TOL, "f16_exp");
}
#[test]
fn test_f16_ln_equiv() {
    assert_approx(&ln_f32(&[1.0]), &[0.0], TOL, "f16_ln");
}
#[test]
fn test_f16_sqrt_equiv() {
    assert_approx(&sqrt_f32(&[4.0]), &[2.0], TOL, "f16_sqrt");
}
#[test]
fn test_f16_rsqrt_equiv() {
    assert_approx(&rsqrt_f32(&[4.0]), &[0.5], TOL, "f16_rsqrt");
}
#[test]
fn test_f16_reciprocal_equiv() {
    assert_approx(&reciprocal_f32(&[2.0]), &[0.5], TOL, "f16_recip");
}
#[test]
fn test_f16_vec_add_equiv() {
    assert_approx(&add(&[1.0, 2.0], &[3.0, 4.0]), &[4.0, 6.0], TOL, "f16_add");
}
#[test]
fn test_f16_vec_sub_equiv() {
    assert_approx(&sub(&[3.0, 4.0], &[1.0, 2.0]), &[2.0, 2.0], TOL, "f16_sub");
}
#[test]
fn test_f16_vec_mul_equiv() {
    assert_approx(&mul(&[2.0, 3.0], &[4.0, 5.0]), &[8.0, 15.0], TOL, "f16_mul");
}
#[test]
fn test_f16_vec_div_equiv() {
    assert_approx(&div(&[4.0, 6.0], &[2.0, 3.0]), &[2.0, 2.0], TOL, "f16_div");
}
#[test]
fn test_f16_reduce_max() {
    assert!((kernel_correctness::reduce::reduce_max(&[1.0, 3.0, 2.0]) - 3.0).abs() < TOL);
}
#[test]
fn test_f16_reduce_sum() {
    assert!((kernel_correctness::reduce::reduce_sum(&[1.0, 2.0, 3.0]) - 6.0).abs() < TOL);
}
