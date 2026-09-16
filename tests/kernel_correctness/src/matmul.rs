//! CPU reference implementations of transposed/triangular matmul kernels.
//! Mirrors: matmul_transpose_kernel.rs

/// C = A^T @ B. A is (k x m), B is (k x n), C is (m x n).
pub fn matmul_transposed_a(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[kk * m + i] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// C = A @ B^T. A is (m x k), B is (n x k), C is (m x n).
pub fn matmul_transposed_b(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[j * k + kk];
            }
            c[i * n + j] = sum;
        }
    }
}

/// C = A^T @ B^T. A is (k x m), B is (n x k), C is (m x n).
pub fn matmul_transposed_both(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[kk * m + i] * b[j * k + kk];
            }
            c[i * n + j] = sum;
        }
    }
}

/// C = tril(A) @ B. Lower triangular: only A[i][k] where k <= i.
pub fn matmul_lower_triangular(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let k_max = (i + 1).min(k);
            for kk in 0..k_max {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// C = triu(A) @ B. Upper triangular: only A[i][k] where k >= i.
pub fn matmul_upper_triangular(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in i..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
