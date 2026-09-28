// build-pass

// Extended optimizer kernels.
// Maps to MultiKernelBench/reference/optimizer/ category (remaining ops).

#![feature(no_core)]

#![no_std]
#![no_core]

/// LAMB optimizer update:
///   m = beta1*m + (1-beta1)*grad
///   v = beta2*v + (1-beta2)*grad^2
///   m_hat = m / (1-beta1^t)
///   v_hat = v / (1-beta2^t)
///   update = m_hat / (sqrt(v_hat) + eps)
///   trust_ratio = ||param|| / ||update|| (if both > 0)
///   param -= lr * trust_ratio * update
/// Maps to optimizer/lamb.py
#[tile_std::tile_kernel]
pub fn lamb_update(
    param: *mut f32, grad: *const f32,
    m_state: *mut f32, v_state: *mut f32,
    config: *const f32, len: *const u32,
) {
    unsafe {
        let n = *len;
        let lr = *config;
        let beta1 = *config.wrapping_add(1);
        let beta2 = *config.wrapping_add(2);
        let eps = *config.wrapping_add(3);
        let beta1_t = *config.wrapping_add(4); // beta1^t (precomputed)
        let beta2_t = *config.wrapping_add(5); // beta2^t (precomputed)

        let inv_1_minus_b1t = 1.0f32 / (1.0f32 - beta1_t);
        let inv_1_minus_b2t = 1.0f32 / (1.0f32 - beta2_t);

        // First pass: update m, v, compute update direction, norms
        let mut param_norm_sq = 0.0f32;
        let mut update_norm_sq = 0.0f32;

        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let g = *grad.wrapping_add(i as usize);
            let p = *(param as *const f32).wrapping_add(i as usize);

            // Update m and v
            let m_old = *(m_state as *const f32).wrapping_add(i as usize);
            let v_old = *(v_state as *const f32).wrapping_add(i as usize);
            let m_new = beta1 * m_old + (1.0f32 - beta1) * g;
            let v_new = beta2 * v_old + (1.0f32 - beta2) * g * g;
            *m_state.wrapping_add(i as usize) = m_new;
            *v_state.wrapping_add(i as usize) = v_new;

            // Bias correction
            let m_hat = m_new * inv_1_minus_b1t;
            let v_hat = v_new * inv_1_minus_b2t;

            // Update direction
            let upd = m_hat / (tile_std::core::builtins::sqrtf(v_hat) + eps);

            // Accumulate norms
            param_norm_sq = param_norm_sq + p * p;
            update_norm_sq = update_norm_sq + upd * upd;

            i = i + 1;
        }

        // Compute trust ratio
        let param_norm = tile_std::core::builtins::sqrtf(param_norm_sq);
        let update_norm = tile_std::core::builtins::sqrtf(update_norm_sq);
        let trust_ratio = if param_norm > 0.0f32 && update_norm > 0.0f32 {
            param_norm / update_norm
        } else {
            1.0f32
        };

        // Second pass: apply update
        i = 0;
        loop {
            if i >= n { break; }
            let m_val = *(m_state as *const f32).wrapping_add(i as usize);
            let v_val = *(v_state as *const f32).wrapping_add(i as usize);
            let m_hat = m_val * inv_1_minus_b1t;
            let v_hat = v_val * inv_1_minus_b2t;
            let upd = m_hat / (tile_std::core::builtins::sqrtf(v_hat) + eps);
            let p = *(param as *const f32).wrapping_add(i as usize);
            *param.wrapping_add(i as usize) = p - lr * trust_ratio * upd;
            i = i + 1;
        }
    }
}
