//! CPU reference implementations of extended optimizer kernels.
//! Mirrors: optimizer_ext_kernel.rs

/// LAMB optimizer update. Returns updated (param, m_state, v_state).
pub fn lamb_update(
    param: &mut [f32],
    grad: &[f32],
    m_state: &mut [f32],
    v_state: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    beta1_t: f32,
    beta2_t: f32,
) {
    let n = param.len();
    let inv_1_minus_b1t = 1.0 / (1.0 - beta1_t);
    let inv_1_minus_b2t = 1.0 / (1.0 - beta2_t);

    // First pass: update m, v; compute norms
    let mut param_norm_sq = 0.0f32;
    let mut update_norm_sq = 0.0f32;

    for i in 0..n {
        let g = grad[i];
        let p = param[i];
        let m_new = beta1 * m_state[i] + (1.0 - beta1) * g;
        let v_new = beta2 * v_state[i] + (1.0 - beta2) * g * g;
        m_state[i] = m_new;
        v_state[i] = v_new;
        let m_hat = m_new * inv_1_minus_b1t;
        let v_hat = v_new * inv_1_minus_b2t;
        let upd = m_hat / (v_hat.sqrt() + eps);
        param_norm_sq += p * p;
        update_norm_sq += upd * upd;
    }

    let param_norm = param_norm_sq.sqrt();
    let update_norm = update_norm_sq.sqrt();
    let trust_ratio = if param_norm > 0.0 && update_norm > 0.0 {
        param_norm / update_norm
    } else {
        1.0
    };

    // Second pass: apply update
    for i in 0..n {
        let m_hat = m_state[i] * inv_1_minus_b1t;
        let v_hat = v_state[i] * inv_1_minus_b2t;
        let upd = m_hat / (v_hat.sqrt() + eps);
        param[i] -= lr * trust_ratio * upd;
    }
}
