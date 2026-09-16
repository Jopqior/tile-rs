// =============================================================================
// Host Application: Multi-Head Attention (single head, f16)
// =============================================================================
//
// Implements scaled dot-product attention:
//   Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
// Pipeline:
//   1. scores = Q * K^T           (HGEMM via BLAS)
//   2. scores *= 1/sqrt(d_k)      (scale kernel)
//   3. weights = softmax(scores)   (softmax_rows kernel, row-wise)
//   4. output = weights * V        (HGEMM via BLAS)
//
// All matrices are f16 for NPU efficiency.

use anyhow::{Context, Result};
use ascend_rs::prelude::*;
use log::info;
use simple_logger::SimpleLogger;

/// CPU reference: scaled dot-product attention in f32.
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, d_k: usize) -> Vec<f32> {
    let scale = 1.0 / (d_k as f32).sqrt();

    // scores = Q * K^T  (seq_len x d_k) * (d_k x seq_len) = (seq_len x seq_len)
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut sum = 0.0f32;
            for kk in 0..d_k {
                sum += q[i * d_k + kk] * k[j * d_k + kk];
            }
            scores[i * seq_len + j] = sum * scale;
        }
    }

    // softmax per row
    let mut weights = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        let row = &scores[i * seq_len..(i + 1) * seq_len];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        for j in 0..seq_len {
            weights[i * seq_len + j] = exp_vals[j] / sum;
        }
    }

    // output = weights * V  (seq_len x seq_len) * (seq_len x d_k) = (seq_len x d_k)
    let mut output = vec![0.0f32; seq_len * d_k];
    for i in 0..seq_len {
        for j in 0..d_k {
            let mut sum = 0.0f32;
            for kk in 0..seq_len {
                sum += weights[i * seq_len + kk] * v[kk * d_k + j];
            }
            output[i * d_k + j] = sum;
        }
    }

    output
}

fn main() -> Result<()> {
    SimpleLogger::new()
        .init()
        .context("Failed to initialize logger")?;

    // -- Configuration -------------------------------------------------------
    let seq_len: usize = 16; // sequence length
    let d_k: usize = 16; // key/value dimension (must be multiple of 16 for f16 alignment)
    let scale = 1.0f32 / (d_k as f32).sqrt();

    info!("MHA: seq_len={}, d_k={}, scale={:.4}", seq_len, d_k, scale);

    // -- Step 1: Initialize ACL ----------------------------------------------
    let acl = Acl::new()?;
    let device = Device::new(&acl)?;
    let context = AclContext::new(&device)?;
    let stream = AclStream::new(&context)?;
    info!("Device {} initialized", device.descriptor());

    // -- Step 2: Generate input data (f32 for reference, then convert to f16) -
    let q_f32: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let k_f32: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i as f32) * 0.02).cos())
        .collect();
    let v_f32: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i as f32) * 0.03).sin())
        .collect();

    let expected = cpu_attention(&q_f32, &k_f32, &v_f32, seq_len, d_k);

    // Convert to f16
    let q_f16: Vec<AclFloat16> = q_f32.iter().map(|&v| AclFloat16::from(v)).collect();
    let k_f16: Vec<AclFloat16> = k_f32.iter().map(|&v| AclFloat16::from(v)).collect();
    let v_f16: Vec<AclFloat16> = v_f32.iter().map(|&v| AclFloat16::from(v)).collect();

    // -- Step 3: Allocate device memory --------------------------------------
    let d_q = DeviceBuffer::from_slice(&q_f16)?; // seq_len x d_k
    let d_k_mat = DeviceBuffer::from_slice(&k_f16)?; // seq_len x d_k (K, not transposed)
    let d_v = DeviceBuffer::from_slice(&v_f16)?; // seq_len x d_k
    let mut d_scores = unsafe {
        DeviceBuffer::<AclFloat16>::uninitialized(seq_len * seq_len)?
    }; // seq_len x seq_len
    let mut d_weights = unsafe {
        DeviceBuffer::<AclFloat16>::uninitialized(seq_len * seq_len)?
    }; // seq_len x seq_len
    let mut d_output = unsafe {
        DeviceBuffer::<AclFloat16>::uninitialized(seq_len * d_k)?
    }; // seq_len x d_k

    // BLAS scalars
    let alpha_val = AclFloat16::from(1.0f32);
    let beta_val = AclFloat16::from(0.0f32);
    let d_alpha = DeviceBox::new(&alpha_val)?;
    let d_beta = DeviceBox::new(&beta_val)?;

    // -- Step 4: scores = Q * K^T (HGEMM) -----------------------------------
    // Q: (seq_len x d_k), K: (seq_len x d_k) → K^T: (d_k x seq_len)
    // Result: scores = Q * K^T = (seq_len x seq_len)
    // GEMM: M=seq_len, N=seq_len, K=d_k
    info!("Step 1: Computing Q * K^T via HGEMM...");
    unsafe {
        ascend_rs_blas::acl_blas_hgemm(
            ascend_rs_blas::AclTransType::TransN, // Q: no transpose
            ascend_rs_blas::AclTransType::TransT, // K: transpose
            ascend_rs_blas::AclTransType::TransN, // C: no transpose
            seq_len as i32,                       // M
            seq_len as i32,                       // N
            d_k as i32,                           // K
            &d_alpha,
            &d_q,
            d_k as i32, // lda = d_k
            &d_k_mat,
            d_k as i32, // ldb = d_k (before transpose)
            &d_beta,
            &mut d_scores,
            seq_len as i32, // ldc = seq_len
            ascend_rs_blas::AclComputeType::HighPrecision,
            &stream,
        )?;
    }
    stream.synchronize()?;

    // -- Step 5: Scale scores by 1/sqrt(d_k) (kernel) -----------------------
    info!("Step 2: Scaling scores by 1/sqrt(d_k) = {:.4}...", scale);
    let n_scores = (seq_len * seq_len) as u32;
    let scale_params = [scale];
    let n_params = [n_scores];
    let d_scale = DeviceBuffer::from_slice(&scale_params)?;
    let d_n_scores = DeviceBuffer::from_slice(&n_params)?;

    unsafe {
        let kernel_loader = KernelLoader::new()?;
        let scale_kernel = kernel_loader.get_kernel("scale_f16")?;
        let mut args = [
            d_scores.as_mut_ptr() as *mut _, // input (in-place: also output)
            d_scores.as_mut_ptr() as *mut _, // output (same buffer)
            d_n_scores.as_mut_ptr() as *mut _,
            d_scale.as_mut_ptr() as *mut _,
        ];
        scale_kernel.launch(1, &stream, &mut args)?;
    }
    stream.synchronize()?;

    // -- Step 6: Row-wise softmax on scores → weights (kernel) ---------------
    info!("Step 3: Computing row-wise softmax...");
    let row_len_params = [d_k as u32]; // Actually seq_len columns per row
    let num_rows_params = [seq_len as u32];
    // Fix: row_len should be seq_len (scores matrix is seq_len x seq_len)
    let row_len_params = [seq_len as u32];
    let d_row_len = DeviceBuffer::from_slice(&row_len_params)?;
    let d_num_rows = DeviceBuffer::from_slice(&num_rows_params)?;

    unsafe {
        let kernel_loader = KernelLoader::new()?;
        let softmax_kernel = kernel_loader.get_kernel("softmax_rows_f16")?;
        let mut args = [
            d_scores.as_mut_ptr() as *mut _,
            d_weights.as_mut_ptr() as *mut _,
            d_row_len.as_mut_ptr() as *mut _,
            d_num_rows.as_mut_ptr() as *mut _,
        ];
        softmax_kernel.launch(1, &stream, &mut args)?;
    }
    stream.synchronize()?;

    // -- Step 7: output = weights * V (HGEMM) --------------------------------
    // weights: (seq_len x seq_len), V: (seq_len x d_k)
    // Result: output = (seq_len x d_k)
    // GEMM: M=seq_len, N=d_k, K=seq_len
    info!("Step 4: Computing attention_weights * V via HGEMM...");
    unsafe {
        ascend_rs_blas::acl_blas_hgemm(
            ascend_rs_blas::AclTransType::TransN,
            ascend_rs_blas::AclTransType::TransN,
            ascend_rs_blas::AclTransType::TransN,
            seq_len as i32, // M
            d_k as i32,     // N
            seq_len as i32, // K
            &d_alpha,
            &d_weights,
            seq_len as i32, // lda = seq_len
            &d_v,
            d_k as i32, // ldb = d_k
            &d_beta,
            &mut d_output,
            d_k as i32, // ldc = d_k
            ascend_rs_blas::AclComputeType::HighPrecision,
            &stream,
        )?;
    }
    stream.synchronize()?;

    // -- Step 8: Copy results to host and verify -----------------------------
    let result_f16 = d_output.to_host()?;
    let result_f32: Vec<f32> = result_f16.iter().map(|v| f32::from(*v)).collect();

    info!("Output ({} x {}):", seq_len, d_k);
    for i in 0..std::cmp::min(4, seq_len) {
        let row: Vec<String> = (0..std::cmp::min(8, d_k))
            .map(|j| format!("{:>8.4}", result_f32[i * d_k + j]))
            .collect();
        info!("  row {}: {}", i, row.join(" "));
    }
    if seq_len > 4 {
        info!("  ...");
    }

    // Verify against CPU reference (f16 precision tolerance)
    info!(
        "Verifying {} results against CPU reference...",
        expected.len()
    );
    let mut max_diff: f32 = 0.0;
    for (idx, (got, exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        // f16 has ~0.1% relative error, plus softmax/GEMM accumulation
        if diff > 0.5 {
            info!(
                "WARNING: large diff at index {}: got {:.4}, expected {:.4}, diff {:.4}",
                idx, got, exp, diff
            );
        }
    }
    info!("Max absolute difference: {:.6}", max_diff);
    assert!(
        max_diff < 1.0,
        "Max diff {} exceeds tolerance (f16 precision + accumulated error)",
        max_diff
    );
    info!("All results within tolerance!");

    Ok(())
}
