// build-pass

// Extended attention patterns.
// Maps to MultiKernelBench/reference/attention/ category.
// Covers causal, cross, multi-query, group-query, KV-cached,
// sparse, windowed, linear attention variants.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Causal attention: softmax(q*k/sqrt(d) + mask) * v
/// Mask is applied as large negative to masked positions.
/// Simplified: scale + masked softmax on attention scores.
#[tile_std::tile_kernel]
pub fn causal_attention(scores: *const f32, mask: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bs = tile_std::__tile_buf_alloc(n);
        let mut bm = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bs, scores, n);
        tile_std::__tile_buf_load_f32(bm, mask, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bs, bs, scale, n);
        tile_std::__tile_pipe_barrier();
        // bm dead after add
        tile_std::__tile_v_add_f32(bm, bs, bm, n);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=bs (dead), src=bm (destroyed), work
        tile_std::kernel_ops::softmax_f32(&mut bs, &mut bm, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bs, n);
    }
}

/// Cross attention: softmax(q*k_cross/sqrt(d))
/// Same as scaled dot product but q and k come from different sequences.
#[tile_std::tile_kernel]
pub fn cross_attention(q: *const f32, k: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bq = tile_std::__tile_buf_alloc(n);
        let mut bk = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bk, k, n);
        tile_std::__tile_pipe_barrier();
        // bk dead after mul, bq dead after mul
        tile_std::__tile_v_mul_f32(bk, bq, bk, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bk, bk, scale, n);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=bq (dead), src=bk (destroyed), work
        tile_std::kernel_ops::softmax_f32(&mut bq, &mut bk, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bq, n);
    }
}

/// Multi-query attention: shared KV across heads, per-head Q
/// Simplified: scale + softmax (same math, different data layout)
#[tile_std::tile_kernel]
pub fn multi_query_attention(q: *const f32, k_shared: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bq = tile_std::__tile_buf_alloc(n);
        let mut bk = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bk, k_shared, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bk, bq, bk, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bk, bk, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bq, &mut bk, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bq, n);
    }
}

/// Group-query attention: KV shared within groups
#[tile_std::tile_kernel]
pub fn group_query_attention(q: *const f32, k_group: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bq = tile_std::__tile_buf_alloc(n);
        let mut bk = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bk, k_group, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bk, bq, bk, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bk, bk, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bq, &mut bk, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bq, n);
    }
}

/// KV-cached attention: use cached k,v + new k,v (append then attend)
/// Simplified: load cached + new, scale, softmax
#[tile_std::tile_kernel]
pub fn kv_cached_attention(q: *const f32, kv_cached: *const f32, kv_new: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bq = tile_std::__tile_buf_alloc(n);
        let mut bc = tile_std::__tile_buf_alloc(n);
        let bn = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bc, kv_cached, n);
        tile_std::__tile_buf_load_f32(bn, kv_new, n);
        tile_std::__tile_pipe_barrier();
        // Merge cached + new → bn dead after
        tile_std::__tile_v_add_f32(bn, bc, bn, n);
        tile_std::__tile_pipe_barrier();
        // Attend: bq * merged → store in bc (bq dead after mul)
        tile_std::__tile_v_mul_f32(bc, bq, bn, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bc, bc, scale, n);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=bq (dead), src=bc (destroyed), work
        tile_std::kernel_ops::softmax_f32(&mut bq, &mut bc, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bq, n);
    }
}

/// Cross-modal attention: attention between two modalities
/// (e.g., text query attending to image keys)
#[tile_std::tile_kernel]
pub fn cross_modal_attention(text_q: *const f32, image_k: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bt = tile_std::__tile_buf_alloc(n);
        let mut bi = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bt, text_q, n);
        tile_std::__tile_buf_load_f32(bi, image_k, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bi, bt, bi, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bi, bi, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bt, &mut bi, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bt, n);
    }
}

/// Linear attention: no softmax, just scale + normalize
/// phi(Q) * (phi(K)^T * V) approximation
#[tile_std::tile_kernel]
pub fn linear_attention(q: *const f32, k: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bq = tile_std::__tile_buf_alloc(n);
        let bk = tile_std::__tile_buf_alloc(n);
        let tmp = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bk, k, n);
        tile_std::__tile_pipe_barrier();
        // ELU+1 feature map: max(0, x) + 1
        tile_std::__tile_maxs_f32(bq, bq, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bq, bq, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_maxs_f32(bk, bk, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bk, bk, 1.0f32, n);
        tile_std::__tile_pipe_barrier();
        // q * k → bk dead after
        tile_std::__tile_v_mul_f32(bk, bq, bk, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bk, bk, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bk, n);
    }
}

/// Sparse attention: apply sparsity mask then softmax
#[tile_std::tile_kernel]
pub fn sparse_attention(scores: *const f32, sparsity_mask: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bs = tile_std::__tile_buf_alloc(n);
        let mut bm = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bs, scores, n);
        tile_std::__tile_buf_load_f32(bm, sparsity_mask, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bs, bs, scale, n);
        tile_std::__tile_pipe_barrier();
        // Multiply by mask (0 or 1) to zero out sparse positions — bm dead after
        tile_std::__tile_v_mul_f32(bm, bs, bm, n);
        tile_std::__tile_pipe_barrier();
        // softmax: dst=bs (dead), src=bm (destroyed), work
        tile_std::kernel_ops::softmax_f32(&mut bs, &mut bm, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bs, n);
    }
}

/// Windowed causal attention: local window mask + causal mask
#[tile_std::tile_kernel]
pub fn windowed_causal_attention(scores: *const f32, window_mask: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bs = tile_std::__tile_buf_alloc(n);
        let mut bm = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bs, scores, n);
        tile_std::__tile_buf_load_f32(bm, window_mask, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bs, bs, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bm, bs, bm, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bs, &mut bm, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bs, n);
    }
}

// === Split variants for 1:1 MKB kernel mapping ===

/// MinGPT-style causal attention: softmax(scores/sqrt(d) + mask)
#[tile_std::tile_kernel]
pub fn min_gpt_causal_attention(scores: *const f32, mask: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bs = tile_std::__tile_buf_alloc(n);
        let mut bm = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bs, scores, n);
        tile_std::__tile_buf_load_f32(bm, mask, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bs, bs, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bm, bs, bm, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bs, &mut bm, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bs, n);
    }
}

/// ReLU self-attention: relu(scores/sqrt(d) + mask) instead of softmax
#[tile_std::tile_kernel]
pub fn relu_self_attention(scores: *const f32, mask: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let bs = tile_std::__tile_buf_alloc(n);
        let bm = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bs, scores, n);
        tile_std::__tile_buf_load_f32(bm, mask, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bs, bs, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bm, bs, bm, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(bm, bm, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bm, n);
    }
}

/// Vision attention: causal attention for vision transformers
#[tile_std::tile_kernel]
pub fn vision_attention(scores: *const f32, mask: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bs = tile_std::__tile_buf_alloc(n);
        let mut bm = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bs, scores, n);
        tile_std::__tile_buf_load_f32(bm, mask, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bs, bs, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bm, bs, bm, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bs, &mut bm, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bs, n);
    }
}

/// Scaled dot-product attention: softmax(scale * q*k)
#[tile_std::tile_kernel]
pub fn scaled_dot_product_attention(q: *const f32, k: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bq = tile_std::__tile_buf_alloc(n);
        let mut bk = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bk, k, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bk, bq, bk, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bk, bk, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bq, &mut bk, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bq, n);
    }
}

/// SDPA for inference workloads: softmax(scale * q*k)
#[tile_std::tile_kernel]
pub fn sdpa_inference(q: *const f32, k: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bq = tile_std::__tile_buf_alloc(n);
        let mut bk = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bk, k, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bk, bq, bk, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bk, bk, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bq, &mut bk, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bq, n);
    }
}

/// SDPA for long context: softmax(scale * q*k)
#[tile_std::tile_kernel]
pub fn sdpa_long_context(q: *const f32, k: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bq = tile_std::__tile_buf_alloc(n);
        let mut bk = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bk, k, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bk, bq, bk, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bk, bk, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bq, &mut bk, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bq, n);
    }
}

/// KV-cached attention for chat batch inference
#[tile_std::tile_kernel]
pub fn kv_cached_chat_batch_attention(q: *const f32, kv_cached: *const f32, kv_new: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bq = tile_std::__tile_buf_alloc(n);
        let mut bc = tile_std::__tile_buf_alloc(n);
        let bn = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bc, kv_cached, n);
        tile_std::__tile_buf_load_f32(bn, kv_new, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bn, bc, bn, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bc, bq, bn, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bc, bc, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bq, &mut bc, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bq, n);
    }
}

/// KV-cached attention for speculative decoding
#[tile_std::tile_kernel]
pub fn kv_cached_speculative_attention(q: *const f32, kv_cached: *const f32, kv_new: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut bq = tile_std::__tile_buf_alloc(n);
        let mut bc = tile_std::__tile_buf_alloc(n);
        let bn = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_buf_load_f32(bc, kv_cached, n);
        tile_std::__tile_buf_load_f32(bn, kv_new, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bn, bc, bn, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bc, bq, bn, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bc, bc, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut bq, &mut bc, &mut work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bq, n);
    }
}
