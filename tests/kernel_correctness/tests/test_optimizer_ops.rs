use kernel_correctness::golden::assert_approx;

const TOL: f32 = 1e-5;

fn sgd_update(param: &[f32], grad: &[f32], lr: f32) -> Vec<f32> {
    param.iter().zip(grad).map(|(&p, &g)| p - lr * g).collect()
}

fn sgd_momentum(
    param: &[f32],
    grad: &[f32],
    velocity: &mut Vec<f32>,
    lr: f32,
    momentum: f32,
) -> Vec<f32> {
    for i in 0..velocity.len() {
        velocity[i] = momentum * velocity[i] + grad[i];
    }
    param
        .iter()
        .zip(velocity.iter())
        .map(|(&p, &v)| p - lr * v)
        .collect()
}

fn adagrad_update(param: &[f32], grad: &[f32], cache: &mut Vec<f32>, lr: f32) -> Vec<f32> {
    let eps = 1e-8f32;
    for i in 0..cache.len() {
        cache[i] += grad[i] * grad[i];
    }
    param
        .iter()
        .zip(grad.iter().zip(cache.iter()))
        .map(|(&p, (&g, &c))| p - lr * g / (c.sqrt() + eps))
        .collect()
}

fn rmsprop_update(
    param: &[f32],
    grad: &[f32],
    cache: &mut Vec<f32>,
    lr: f32,
    decay: f32,
) -> Vec<f32> {
    let eps = 1e-8f32;
    for i in 0..cache.len() {
        cache[i] = decay * cache[i] + (1.0 - decay) * grad[i] * grad[i];
    }
    param
        .iter()
        .zip(grad.iter().zip(cache.iter()))
        .map(|(&p, (&g, &c))| p - lr * g / (c.sqrt() + eps))
        .collect()
}

fn adam_update(
    param: &[f32],
    grad: &[f32],
    m: &mut Vec<f32>,
    v: &mut Vec<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
) -> Vec<f32> {
    let eps = 1e-8f32;
    for i in 0..m.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
    }
    param
        .iter()
        .zip(m.iter().zip(v.iter()))
        .map(|(&p, (&mi, &vi))| p - lr * mi / (vi.sqrt() + eps))
        .collect()
}

#[test]
fn test_sgd_basic() {
    assert_approx(
        &sgd_update(&[1.0, 2.0], &[0.1, 0.2], 0.01),
        &[0.999, 1.998],
        TOL,
        "sgd",
    );
}

#[test]
fn test_sgd_momentum_basic() {
    let mut vel = vec![0.0, 0.0];
    let result = sgd_momentum(&[1.0, 2.0], &[0.1, 0.2], &mut vel, 0.01, 0.9);
    assert_approx(&result, &[0.999, 1.998], TOL, "sgd_mom");
}

#[test]
fn test_adagrad_basic() {
    let mut cache = vec![0.0, 0.0];
    let result = adagrad_update(&[1.0, 2.0], &[0.1, 0.2], &mut cache, 0.01);
    // cache becomes [0.01, 0.04], update = lr * g / (sqrt(cache) + eps)
    let expected_0 = 1.0 - 0.01 * 0.1 / (0.01f32.sqrt() + 1e-8);
    let expected_1 = 2.0 - 0.01 * 0.2 / (0.04f32.sqrt() + 1e-8);
    assert_approx(&result, &[expected_0, expected_1], TOL, "adagrad");
}

#[test]
fn test_rmsprop_basic() {
    let mut cache = vec![0.0, 0.0];
    let result = rmsprop_update(&[1.0, 2.0], &[0.1, 0.2], &mut cache, 0.01, 0.9);
    assert!(
        result[0] < 1.0 && result[1] < 2.0,
        "rmsprop decreases params"
    );
}

#[test]
fn test_adam_basic() {
    let mut m = vec![0.0, 0.0];
    let mut v = vec![0.0, 0.0];
    let result = adam_update(&[1.0, 2.0], &[0.1, 0.2], &mut m, &mut v, 0.001, 0.9, 0.999);
    assert!(result[0] < 1.0 && result[1] < 2.0, "adam decreases params");
}
