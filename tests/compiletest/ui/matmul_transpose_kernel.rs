// build-pass

// Matrix multiply kernels with transpose and triangular masking.
// Maps to MultiKernelBench/reference/matmul/ category.
// Uses scalar loops for transpose/masking since cube engine
// doesn't natively support transposed inputs.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Matmul with A transposed: C[i][j] = sum_k A[k][i] * B[k][j]
/// Maps to matmul/matmul_transposed_a.py
#[tile_std::tile_kernel]
pub fn matmul_transposed_a(
    a: *const f32, b: *const f32, c: *mut f32, dims: *const u32,
) {
    unsafe {
        let m = *dims;        // rows of C (= cols of A)
        let k = *dims.wrapping_add(1); // shared dim (= rows of A = rows of B)
        let n = *dims.wrapping_add(2); // cols of C (= cols of B)

        let mut i = 0u32;
        loop {
            if i >= m { break; }
            let mut j = 0u32;
            loop {
                if j >= n { break; }
                let mut sum = 0.0f32;
                let mut kk = 0u32;
                loop {
                    if kk >= k { break; }
                    // A^T[i][kk] = A[kk][i]
                    let a_val = *a.wrapping_add((kk * m + i) as usize);
                    let b_val = *b.wrapping_add((kk * n + j) as usize);
                    sum = sum + a_val * b_val;
                    kk = kk + 1;
                }
                *c.wrapping_add((i * n + j) as usize) = sum;
                j = j + 1;
            }
            i = i + 1;
        }
    }
}

/// Matmul with B transposed: C[i][j] = sum_k A[i][k] * B[j][k]
/// Maps to matmul/matmul_transposed_b.py
#[tile_std::tile_kernel]
pub fn matmul_transposed_b(
    a: *const f32, b: *const f32, c: *mut f32, dims: *const u32,
) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        let mut i = 0u32;
        loop {
            if i >= m { break; }
            let mut j = 0u32;
            loop {
                if j >= n { break; }
                let mut sum = 0.0f32;
                let mut kk = 0u32;
                loop {
                    if kk >= k { break; }
                    let a_val = *a.wrapping_add((i * k + kk) as usize);
                    // B^T[kk][j] = B[j][kk]
                    let b_val = *b.wrapping_add((j * k + kk) as usize);
                    sum = sum + a_val * b_val;
                    kk = kk + 1;
                }
                *c.wrapping_add((i * n + j) as usize) = sum;
                j = j + 1;
            }
            i = i + 1;
        }
    }
}

/// Matmul with both A and B transposed: C[i][j] = sum_k A[k][i] * B[j][k]
/// Maps to matmul/matmul_transposed_both.py
#[tile_std::tile_kernel]
pub fn matmul_transposed_both(
    a: *const f32, b: *const f32, c: *mut f32, dims: *const u32,
) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        let mut i = 0u32;
        loop {
            if i >= m { break; }
            let mut j = 0u32;
            loop {
                if j >= n { break; }
                let mut sum = 0.0f32;
                let mut kk = 0u32;
                loop {
                    if kk >= k { break; }
                    let a_val = *a.wrapping_add((kk * m + i) as usize);
                    let b_val = *b.wrapping_add((j * k + kk) as usize);
                    sum = sum + a_val * b_val;
                    kk = kk + 1;
                }
                *c.wrapping_add((i * n + j) as usize) = sum;
                j = j + 1;
            }
            i = i + 1;
        }
    }
}

/// Lower triangular matmul: C = tril(A) * B
/// Only uses elements A[i][k] where k <= i.
/// Maps to matmul/matmul_lower_triangular.py
#[tile_std::tile_kernel]
pub fn matmul_lower_triangular(
    a: *const f32, b: *const f32, c: *mut f32, dims: *const u32,
) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        let mut i = 0u32;
        loop {
            if i >= m { break; }
            let mut j = 0u32;
            loop {
                if j >= n { break; }
                let mut sum = 0.0f32;
                // Only sum over k-indices where kk <= i (lower triangular)
                let k_max = if i + 1 < k { i + 1 } else { k };
                let mut kk = 0u32;
                loop {
                    if kk >= k_max { break; }
                    let a_val = *a.wrapping_add((i * k + kk) as usize);
                    let b_val = *b.wrapping_add((kk * n + j) as usize);
                    sum = sum + a_val * b_val;
                    kk = kk + 1;
                }
                *c.wrapping_add((i * n + j) as usize) = sum;
                j = j + 1;
            }
            i = i + 1;
        }
    }
}

/// Upper triangular matmul: C = triu(A) * B
/// Only uses elements A[i][k] where k >= i.
/// Maps to matmul/matmul_upper_triangular.py
#[tile_std::tile_kernel]
pub fn matmul_upper_triangular(
    a: *const f32, b: *const f32, c: *mut f32, dims: *const u32,
) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);

        let mut i = 0u32;
        loop {
            if i >= m { break; }
            let mut j = 0u32;
            loop {
                if j >= n { break; }
                let mut sum = 0.0f32;
                // Only sum over k-indices where kk >= i (upper triangular)
                let mut kk = i;
                loop {
                    if kk >= k { break; }
                    let a_val = *a.wrapping_add((i * k + kk) as usize);
                    let b_val = *b.wrapping_add((kk * n + j) as usize);
                    sum = sum + a_val * b_val;
                    kk = kk + 1;
                }
                *c.wrapping_add((i * n + j) as usize) = sum;
                j = j + 1;
            }
            i = i + 1;
        }
    }
}
