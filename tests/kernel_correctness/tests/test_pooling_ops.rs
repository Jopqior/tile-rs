const TOL: f32 = 1e-5;

fn global_avg_pool(x: &[f32]) -> f32 {
    x.iter().sum::<f32>() / x.len() as f32
}
fn global_max_pool(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}
fn global_min_pool(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::INFINITY, f32::min)
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn lp_pool_2(x: &[f32]) -> f32 {
    (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

// ===== pooling_ops_kernel.rs (6 kernels) =====
#[test]
fn test_global_avg_pool() {
    assert!((global_avg_pool(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < TOL);
}
#[test]
fn test_global_max_pool() {
    assert!((global_max_pool(&[1.0, 5.0, 3.0, 2.0]) - 5.0).abs() < TOL);
}
#[test]
fn test_global_min_pool() {
    assert!((global_min_pool(&[1.0, 5.0, 3.0, 2.0]) - 1.0).abs() < TOL);
}
#[test]
fn test_fused_avgpool_sigmoid() {
    let mean = global_avg_pool(&[1.0, 2.0, 3.0, 4.0]);
    let result = sigmoid(mean);
    assert!((result - sigmoid(2.5)).abs() < TOL, "avgpool_sigmoid");
}
#[test]
fn test_fused_pool_sigmoid_sum() {
    let x = [0.0, 1.0, -1.0, 2.0];
    let sig: Vec<f32> = x.iter().map(|&v| sigmoid(v)).collect();
    let sum: f32 = sig.iter().sum();
    assert!(sum > 0.0, "pool_sigmoid_sum positive");
}
#[test]
fn test_lp_pool_2() {
    assert!(
        (lp_pool_2(&[3.0, 4.0]) - (12.5f32).sqrt()).abs() < TOL,
        "lp_pool_2"
    );
}
