const TOL: f32 = 1e-4;

fn mse_loss(pred: &[f32], target: &[f32]) -> f32 {
    let n = pred.len() as f32;
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum::<f32>()
        / n
}

fn huber_loss(pred: &[f32], target: &[f32], delta: f32) -> f32 {
    let n = pred.len() as f32;
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| {
            let diff = (p - t).abs();
            if diff <= delta {
                0.5 * diff * diff
            } else {
                delta * (diff - 0.5 * delta)
            }
        })
        .sum::<f32>()
        / n
}

fn hinge_loss(pred: &[f32], target: &[f32]) -> f32 {
    let n = pred.len() as f32;
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| (1.0 - p * t).max(0.0))
        .sum::<f32>()
        / n
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn cross_entropy_loss(pred: &[f32], target: &[f32]) -> f32 {
    let n = pred.len() as f32;
    -pred
        .iter()
        .zip(target)
        .map(|(&p, &t)| t * p.ln())
        .sum::<f32>()
        / n
}

fn kl_div_loss(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q)
        .map(|(&pi, &qi)| pi * (pi.ln() - qi.ln()))
        .sum()
}

// ===== loss_ops_kernel.rs (6 kernels) =====

#[test]
fn test_mse_loss_zero() {
    assert!((mse_loss(&[1.0, 2.0], &[1.0, 2.0])).abs() < TOL);
}
#[test]
fn test_mse_loss_basic() {
    assert!((mse_loss(&[1.0, 2.0], &[3.0, 4.0]) - 4.0).abs() < TOL);
}
#[test]
fn test_huber_loss_small() {
    assert!((huber_loss(&[1.0, 2.0], &[1.1, 2.1], 1.0) - 0.005).abs() < TOL);
}
#[test]
fn test_huber_loss_large() {
    assert!((huber_loss(&[1.0], &[5.0], 1.0) - 3.5).abs() < TOL);
}
#[test]
fn test_hinge_loss_correct() {
    assert!((hinge_loss(&[2.0], &[1.0])).abs() < TOL);
}
#[test]
fn test_hinge_loss_wrong() {
    assert!((hinge_loss(&[-1.0], &[1.0]) - 2.0).abs() < TOL);
}
#[test]
fn test_cosine_similarity_same() {
    assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < TOL);
}
#[test]
fn test_cosine_similarity_ortho() {
    assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < TOL);
}
#[test]
fn test_cross_entropy() {
    let pred = [0.7, 0.2, 0.1];
    let target = [1.0, 0.0, 0.0];
    let ce = cross_entropy_loss(&pred, &target);
    assert!((ce - (-0.7f32.ln()) / 3.0).abs() < TOL, "ce: {ce}");
}
#[test]
fn test_kl_div() {
    let p = [0.5, 0.5];
    let q = [0.5, 0.5];
    assert!((kl_div_loss(&p, &q)).abs() < TOL, "kl_same");
}
