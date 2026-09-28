use kernel_correctness::matmul;

const TOL: f32 = 1e-5;

fn assert_approx(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(err < TOL, "{ctx} elem {i}: got {g}, want {w}, err {err}");
    }
}

// ====================================================================
// Transposed A: C = A^T @ B
// ====================================================================

#[test]
fn transposed_a_identity() {
    // A = I (2x2), B = [1,2; 3,4], C = I^T @ B = B
    let a = [1.0, 0.0, 0.0, 1.0]; // (k=2 x m=2) row-major
    let b = [1.0, 2.0, 3.0, 4.0]; // (k=2 x n=2)
    let mut c = [0.0; 4];
    matmul::matmul_transposed_a(&a, &b, &mut c, 2, 2, 2);
    assert_approx(&c, &[1.0, 2.0, 3.0, 4.0], "transA_identity");
}

#[test]
fn transposed_a_manual() {
    // A stored as (k=2 x m=3): [[1,2,3],[4,5,6]] -> A^T = [[1,4],[2,5],[3,6]]
    // B = (k=2 x n=2): [[1,0],[0,1]]
    // C = A^T @ B = [[1,4],[2,5],[3,6]]
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0, 0.0, 0.0, 1.0];
    let mut c = [0.0; 6]; // m=3, n=2
    matmul::matmul_transposed_a(&a, &b, &mut c, 3, 2, 2);
    assert_approx(&c, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], "transA_manual");
}

// ====================================================================
// Transposed B: C = A @ B^T
// ====================================================================

#[test]
fn transposed_b_identity() {
    let a = [1.0, 2.0, 3.0, 4.0]; // (m=2 x k=2)
    let b = [1.0, 0.0, 0.0, 1.0]; // (n=2 x k=2) -> B^T = I
    let mut c = [0.0; 4];
    matmul::matmul_transposed_b(&a, &b, &mut c, 2, 2, 2);
    assert_approx(&c, &[1.0, 2.0, 3.0, 4.0], "transB_identity");
}

#[test]
fn transposed_b_dot_product() {
    // A = [[1,2,3]], B = [[4,5,6]] (both 1xk=3)
    // C = A @ B^T = dot(A, B) = 1*4+2*5+3*6 = 32
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let mut c = [0.0; 1];
    matmul::matmul_transposed_b(&a, &b, &mut c, 1, 3, 1);
    assert_approx(&c, &[32.0], "transB_dot");
}

// ====================================================================
// Transposed both: C = A^T @ B^T
// ====================================================================

#[test]
fn transposed_both_1x1() {
    let a = [3.0]; // (k=1 x m=1)
    let b = [5.0]; // (n=1 x k=1)
    let mut c = [0.0; 1];
    matmul::matmul_transposed_both(&a, &b, &mut c, 1, 1, 1);
    assert_approx(&c, &[15.0], "transBoth_1x1");
}

#[test]
fn transposed_both_manual() {
    // A stored as (k=2 x m=2): [[1,3],[2,4]] -> A^T = [[1,2],[3,4]]
    // B stored as (n=2 x k=2): [[5,6],[7,8]] -> B^T = [[5,7],[6,8]]
    // C = A^T @ B^T = [[1,2],[3,4]] @ [[5,7],[6,8]] = [[17,23],[39,53]]
    let a = [1.0, 3.0, 2.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let mut c = [0.0; 4];
    matmul::matmul_transposed_both(&a, &b, &mut c, 2, 2, 2);
    assert_approx(&c, &[17.0, 23.0, 39.0, 53.0], "transBoth_manual");
}

// ====================================================================
// Lower triangular
// ====================================================================

#[test]
fn lower_triangular_identity() {
    // tril(I) = I
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3x3 identity
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut c = [0.0; 9];
    matmul::matmul_lower_triangular(&a, &b, &mut c, 3, 3, 3);
    assert_approx(&c, &b, "lower_tri_identity");
}

#[test]
fn lower_triangular_3x3() {
    // A = [[1,0,0],[2,3,0],[4,5,6]] (lower triangular)
    // B = [[1],[1],[1]]
    // C = [[1],[2+3],[4+5+6]] = [[1],[5],[15]]
    let a = [1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0];
    let b = [1.0, 1.0, 1.0];
    let mut c = [0.0; 3];
    matmul::matmul_lower_triangular(&a, &b, &mut c, 3, 3, 1);
    assert_approx(&c, &[1.0, 5.0, 15.0], "lower_tri_3x3");
}

// ====================================================================
// Upper triangular
// ====================================================================

#[test]
fn upper_triangular_identity() {
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut c = [0.0; 9];
    matmul::matmul_upper_triangular(&a, &b, &mut c, 3, 3, 3);
    assert_approx(&c, &b, "upper_tri_identity");
}

#[test]
fn upper_triangular_3x3() {
    // A = [[1,2,3],[0,4,5],[0,0,6]] (upper triangular)
    // B = [[1],[1],[1]]
    // C = [[1+2+3],[4+5],[6]] = [[6],[9],[6]]
    let a = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0];
    let b = [1.0, 1.0, 1.0];
    let mut c = [0.0; 3];
    matmul::matmul_upper_triangular(&a, &b, &mut c, 3, 3, 1);
    assert_approx(&c, &[6.0, 9.0, 6.0], "upper_tri_3x3");
}

#[test]
fn upper_lower_complementary() {
    // For a general matrix A and identity-like B:
    // tril(A) @ B + triu_strict(A) @ B should equal A @ B
    // But with our definition (triu includes diagonal), check:
    // tril includes diagonal, triu includes diagonal
    // So their sum double-counts diagonal.
    let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
    let b = [1.0, 0.0, 0.0, 1.0]; // identity
    let mut cl = [0.0; 4];
    let mut cu = [0.0; 4];
    matmul::matmul_lower_triangular(&a, &b, &mut cl, 2, 2, 2);
    matmul::matmul_upper_triangular(&a, &b, &mut cu, 2, 2, 2);
    // lower: uses a[i][k] for k<=i -> C_lower = [[1,0],[3,4]]  (only a[0][0], a[1][0], a[1][1])
    // upper: uses a[i][k] for k>=i -> C_upper = [[1,2],[0,4]]  (only a[0][0], a[0][1], a[1][1])
    assert_approx(&cl, &[1.0, 0.0, 3.0, 4.0], "comp_lower");
    assert_approx(&cu, &[1.0, 2.0, 0.0, 4.0], "comp_upper");
}
