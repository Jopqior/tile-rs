//! CPU reference implementations of extended loss kernels.
//! Mirrors: loss_ext_kernel.rs

/// Triplet margin loss: max(0, ||a-p||^2 - ||a-n||^2 + margin)
pub fn triplet_margin_loss(anchor: &[f32], positive: &[f32], negative: &[f32], margin: f32) -> f32 {
    let mut dist_ap = 0.0f32;
    let mut dist_an = 0.0f32;
    for i in 0..anchor.len() {
        let dp = anchor[i] - positive[i];
        dist_ap += dp * dp;
        let dn = anchor[i] - negative[i];
        dist_an += dn * dn;
    }
    (dist_ap - dist_an + margin).max(0.0)
}
