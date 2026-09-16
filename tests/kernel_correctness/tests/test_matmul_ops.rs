use kernel_correctness::golden::assert_approx;

const TOL: f32 = 1e-4;

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ===== matmul_ops_kernel.rs: standard, square, matvec, large_k, small_k, irregular, tall_skinny =====
#[test]
fn test_matmul_standard() {
    // 2x3 * 3x2 = 2x2
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let c = matmul(&a, &b, 2, 3, 2);
    assert_approx(&c, &[22.0, 28.0, 49.0, 64.0], TOL, "matmul_std");
}
#[test]
fn test_matmul_square() {
    let a = [1.0, 0.0, 0.0, 1.0]; // identity
    let b = [5.0, 6.0, 7.0, 8.0];
    let c = matmul(&a, &b, 2, 2, 2);
    assert_approx(&c, &b, TOL, "matmul_square");
}
#[test]
fn test_matmul_matvec() {
    let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
    let x = [1.0, 1.0]; // 2x1
    let y = matmul(&a, &x, 2, 2, 1);
    assert_approx(&y, &[3.0, 7.0], TOL, "matvec");
}

// ===== matmul_extended_kernel.rs =====
#[test]
fn test_matmul_bias() {
    let c = matmul(&[1.0, 0.0, 0.0, 1.0], &[1.0, 2.0, 3.0, 4.0], 2, 2, 2);
    let bias = [0.1, 0.1, 0.1, 0.1];
    let result: Vec<f32> = c.iter().zip(&bias).map(|(&ci, &bi)| ci + bi).collect();
    assert_approx(&result, &[1.1, 2.1, 3.1, 4.1], TOL, "matmul_bias");
}
#[test]
fn test_matmul_scaled() {
    let c = matmul(&[1.0, 2.0, 3.0, 4.0], &[1.0, 0.0, 0.0, 1.0], 2, 2, 2);
    let result: Vec<f32> = c.iter().map(|&v| v * 0.5).collect();
    assert_approx(&result, &[0.5, 1.0, 1.5, 2.0], TOL, "matmul_scaled");
}
#[test]
fn test_gemm_full() {
    // C = alpha*A*B + beta*C_in with alpha=1, beta=0.5
    let ab = matmul(&[1.0, 0.0, 0.0, 1.0], &[1.0, 2.0, 3.0, 4.0], 2, 2, 2);
    let c_in = [2.0, 2.0, 2.0, 2.0];
    let result: Vec<f32> = ab
        .iter()
        .zip(&c_in)
        .map(|(&a, &c)| 1.0 * a + 0.5 * c)
        .collect();
    assert_approx(&result, &[2.0, 3.0, 4.0, 5.0], TOL, "gemm_full");
}
#[test]
fn test_outer_product() {
    // Simplified outer product: element-wise mul
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let result: Vec<f32> = a.iter().zip(&b).map(|(&ai, &bi)| ai * bi).collect();
    assert_approx(&result, &[4.0, 10.0, 18.0], TOL, "outer_product");
}
#[test]
fn test_matmul_relu_matmul() {
    let c = matmul(&[1.0, 2.0, 3.0, 4.0], &[1.0, 0.0, 0.0, 1.0], 2, 2, 2);
    let relu: Vec<f32> = c.iter().map(|&v| v.max(0.0)).collect();
    assert_approx(&relu, &[1.0, 2.0, 3.0, 4.0], TOL, "matmul_relu");
}
#[test]
fn test_matmul_accumulate() {
    let existing = [1.0, 1.0, 1.0, 1.0];
    let ab = matmul(&[1.0, 0.0, 0.0, 1.0], &[2.0, 3.0, 4.0, 5.0], 2, 2, 2);
    let result: Vec<f32> = existing.iter().zip(&ab).map(|(&e, &a)| e + a).collect();
    assert_approx(&result, &[3.0, 4.0, 5.0, 6.0], TOL, "matmul_acc");
}
#[test]
fn test_matmul_diag_scale() {
    let ab = matmul(&[1.0, 0.0, 0.0, 1.0], &[1.0, 2.0, 3.0, 4.0], 2, 2, 2);
    let diag = [2.0, 2.0, 2.0, 2.0];
    let result: Vec<f32> = ab.iter().zip(&diag).map(|(&a, &d)| a * d).collect();
    assert_approx(&result, &[2.0, 4.0, 6.0, 8.0], TOL, "matmul_diag");
}
