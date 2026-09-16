// build-pass

// Loss function kernels.
// Maps to MultiKernelBench/reference/loss/ category.

#![feature(no_core)]

#![no_std]
#![no_core]

/// MSE Loss: mse(pred, target) = mean((pred - target)^2)
/// Maps to loss/mse_loss.py
#[tile_std::tile_kernel]
pub fn mse_loss(pred: *const f32, target: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bp = tile_std::__tile_buf_alloc(n);
        let bt = tile_std::__tile_buf_alloc(n);
        let mut bw = tile_std::__tile_buf_alloc(n);
        let mut btmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, pred, n);
        tile_std::__tile_buf_load_f32(bt, target, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::kernel_ops::mse_loss_f32(&mut bw, &bp, &bt, &mut btmp, n);

        // Broadcast scalar to buffer + DMA store (scalar GM writes don't work on 310P)
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bw, bw, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bw, bw, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bw, n);
    }
}

/// Huber Loss
/// Maps to loss/huber_loss.py
#[tile_std::tile_kernel]
pub fn huber_loss(pred: *const f32, target: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let delta = 1.0f32;
        let mut bp = tile_std::__tile_buf_alloc(n);
        let bt = tile_std::__tile_buf_alloc(n);
        let mut bw = tile_std::__tile_buf_alloc(n);
        let mut btmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, pred, n);
        tile_std::__tile_buf_load_f32(bt, target, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::kernel_ops::huber_loss_f32(&mut bw, &mut bp, &bt, &mut btmp, delta, n);

        // Broadcast scalar to buffer + DMA store (scalar GM writes don't work on 310P)
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bw, bw, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bw, bw, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bw, n);
    }
}

/// Hinge Loss: hinge(pred, target) = mean(max(0, 1 - pred * target))
/// Maps to loss/hinge_loss.py
#[tile_std::tile_kernel]
pub fn hinge_loss(pred: *const f32, target: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bp = tile_std::__tile_buf_alloc(n);
        let bt = tile_std::__tile_buf_alloc(n);
        let mut bw = tile_std::__tile_buf_alloc(n);
        let mut btmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, pred, n);
        tile_std::__tile_buf_load_f32(bt, target, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::kernel_ops::hinge_loss_f32(&mut bw, &bp, &bt, &mut btmp, n);

        // Broadcast scalar to buffer + DMA store (scalar GM writes don't work on 310P)
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bw, bw, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bw, bw, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bw, n);
    }
}

/// Cosine Similarity Loss: cos_sim(a, b) = dot(a,b) / (norm(a)*norm(b))
/// Maps to loss/cosine_similarity_loss.py
#[tile_std::tile_kernel]
pub fn cosine_similarity(a: *const f32, b: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let ba = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);
        let mut bw = tile_std::__tile_buf_alloc(n);
        let mut btmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(ba, a, n);
        tile_std::__tile_buf_load_f32(bb, b, n);
        tile_std::__tile_pipe_barrier();

        let result = tile_std::kernel_ops::cosine_similarity_f32(&mut bw, &ba, &bb, &mut btmp, n);

        // Broadcast scalar to buffer + DMA store (scalar GM writes don't work on 310P)
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bw, bw, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bw, bw, result, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bw, n);
    }
}

/// Cross Entropy Loss: ce(pred, target) = -sum(target * log(pred)) / n
/// Maps to loss/cross_entropy_loss.py (simplified, assumes pred is already probabilities)
#[tile_std::tile_kernel]
pub fn cross_entropy_loss(pred: *const f32, target: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bp = tile_std::__tile_buf_alloc(n);
        let bt = tile_std::__tile_buf_alloc(n);
        let bw = tile_std::__tile_buf_alloc(n);
        let btmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, pred, n);
        tile_std::__tile_buf_load_f32(bt, target, n);
        tile_std::__tile_pipe_barrier();

        // log(pred)
        tile_std::__tile_ln_f32(bw, bp, n);
        tile_std::__tile_pipe_barrier();
        // btmp = target * log(pred) — use btmp as output to avoid Mul aliasing
        tile_std::__tile_v_mul_f32(btmp, bt, bw, n);
        tile_std::__tile_pipe_barrier();
        // -sum(target * log(pred))
        let sum = tile_std::__tile_v_reduce_sum_f32(btmp, btmp, bw, n);
        let loss = -sum / (n as f32);

        // Broadcast scalar to buffer + DMA store (scalar GM writes don't work on 310P)
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bw, bw, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bw, bw, loss, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bw, n);
    }
}

/// KL Divergence Loss: kl(p, q) = sum(p * (log(p) - log(q)))
/// Maps to loss/kl_div_loss.py
#[tile_std::tile_kernel]
pub fn kl_div_loss(p: *const f32, q: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bp = tile_std::__tile_buf_alloc(n);
        let bq = tile_std::__tile_buf_alloc(n);
        let bw = tile_std::__tile_buf_alloc(n);
        let btmp = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, p, n);
        tile_std::__tile_buf_load_f32(bq, q, n);
        tile_std::__tile_pipe_barrier();

        // bw = log(p)
        tile_std::__tile_ln_f32(bw, bp, n);
        tile_std::__tile_pipe_barrier();
        // btmp = log(q)
        tile_std::__tile_ln_f32(btmp, bq, n);
        tile_std::__tile_pipe_barrier();
        // bq = log(p) - log(q) — all separate (bq no longer needed after ln)
        tile_std::__tile_v_sub_f32(bq, bw, btmp, n);
        tile_std::__tile_pipe_barrier();
        // bw = p * (log(p) - log(q)) — all separate
        tile_std::__tile_v_mul_f32(bw, bp, bq, n);
        tile_std::__tile_pipe_barrier();
        // sum
        let sum = tile_std::__tile_v_reduce_sum_f32(bw, bw, btmp, n);

        // Broadcast scalar to buffer + DMA store (scalar GM writes don't work on 310P)
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(bw, bw, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bw, bw, sum, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bw, n);
    }
}
