// build-pass

// Optimizer update kernels.
// Maps to MultiKernelBench/reference/optimizer/ category.

#![feature(no_core)]

#![no_std]
#![no_core]

/// SGD update: param = param - lr * grad
/// Maps to optimizer/sgd.py
#[tile_std::tile_kernel]
pub fn sgd_update(param: *mut f32, grad: *const f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let lr = *config;
        let mut bp = tile_std::__tile_buf_alloc(n);
        let mut bg = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, param as *const f32, n);
        tile_std::__tile_buf_load_f32(bg, grad, n);
        tile_std::__tile_pipe_barrier();

        tile_std::kernel_ops::sgd_update_f32(&mut bp, &mut bg, lr, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(param, bp, n);
    }
}

/// SGD with momentum: v = momentum * v + grad; param = param - lr * v
/// Maps to optimizer/sgd.py (with momentum variant)
#[tile_std::tile_kernel]
pub fn sgd_momentum(param: *mut f32, grad: *const f32, velocity: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let lr = *config;
        let momentum = *config.wrapping_add(1);

        let bp = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        let bv = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, param as *const f32, n);
        tile_std::__tile_buf_load_f32(bg, grad, n);
        tile_std::__tile_buf_load_f32(bv, velocity as *const f32, n);
        tile_std::__tile_pipe_barrier();

        // v = momentum * v
        tile_std::__tile_muls_f32(bv, bv, momentum, n);
        tile_std::__tile_pipe_barrier();
        // v = momentum * v + grad → store in bg (dead after), bg = new_v
        tile_std::__tile_v_add_f32(bg, bv, bg, n);
        tile_std::__tile_pipe_barrier();
        // param = param - lr * new_v → bv = lr * new_v (temp)
        tile_std::__tile_muls_f32(bv, bg, lr, n);
        tile_std::__tile_pipe_barrier();
        // bp - bv → store in bv (bv is temp, dead after)
        tile_std::__tile_v_sub_f32(bv, bp, bv, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(param, bv, n);
        tile_std::__tile_buf_store_f32(velocity, bg, n);
    }
}

/// Adagrad update: cache += grad^2; param -= lr * grad / (sqrt(cache) + eps)
/// Maps to optimizer/adagrad.py
#[tile_std::tile_kernel]
pub fn adagrad_update(param: *mut f32, grad: *const f32, cache: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let lr = *config;
        let eps = 1e-8f32;

        let bp = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        let bc = tile_std::__tile_buf_alloc(n);
        let bt = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, param as *const f32, n);
        tile_std::__tile_buf_load_f32(bg, grad, n);
        tile_std::__tile_buf_load_f32(bc, cache as *const f32, n);
        tile_std::__tile_pipe_barrier();

        // bt = grad^2
        tile_std::__tile_v_mul_f32(bt, bg, bg, n);
        tile_std::__tile_pipe_barrier();
        // cache += grad^2 → bt dead (temp), output to bt
        tile_std::__tile_v_add_f32(bt, bc, bt, n);
        // bt now = new cache value
        tile_std::__tile_pipe_barrier();
        // bc = sqrt(cache) + eps (reuse bc as temp)
        tile_std::__tile_v_sqrt_f32(bc, bt, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bc, bc, eps, n);
        tile_std::__tile_pipe_barrier();
        // bc = grad / (sqrt(cache) + eps)
        tile_std::__tile_v_div_f32(bc, bg, bc, n);
        tile_std::__tile_pipe_barrier();
        // bc = lr * grad / (sqrt(cache) + eps)
        tile_std::__tile_muls_f32(bc, bc, lr, n);
        tile_std::__tile_pipe_barrier();
        // param -= update → bc dead after
        tile_std::__tile_v_sub_f32(bc, bp, bc, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(param, bc, n);
        tile_std::__tile_buf_store_f32(cache, bt, n);
    }
}

/// RMSprop update: cache = decay * cache + (1-decay) * grad^2;
///                 param -= lr * grad / (sqrt(cache) + eps)
/// Maps to optimizer/rmsprop.py
#[tile_std::tile_kernel]
pub fn rmsprop_update(param: *mut f32, grad: *const f32, cache: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let lr = *config;
        let decay = *config.wrapping_add(1);
        let eps = 1e-8f32;

        let bp = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        let bc = tile_std::__tile_buf_alloc(n);
        let bt = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, param as *const f32, n);
        tile_std::__tile_buf_load_f32(bg, grad, n);
        tile_std::__tile_buf_load_f32(bc, cache as *const f32, n);
        tile_std::__tile_pipe_barrier();

        // cache = decay * cache
        tile_std::__tile_muls_f32(bc, bc, decay, n);
        // bt = grad^2
        tile_std::__tile_v_mul_f32(bt, bg, bg, n);
        tile_std::__tile_pipe_barrier();
        // bt = (1-decay) * grad^2
        tile_std::__tile_muls_f32(bt, bt, 1.0f32 - decay, n);
        tile_std::__tile_pipe_barrier();
        // cache = decay * cache + (1-decay) * grad^2 → bt = new cache
        tile_std::__tile_v_add_f32(bt, bc, bt, n);
        tile_std::__tile_pipe_barrier();

        // bc = sqrt(cache) + eps
        tile_std::__tile_v_sqrt_f32(bc, bt, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bc, bc, eps, n);
        tile_std::__tile_pipe_barrier();
        // bc = grad / (sqrt(cache) + eps)
        tile_std::__tile_v_div_f32(bc, bg, bc, n);
        tile_std::__tile_pipe_barrier();
        // bc = lr * ...
        tile_std::__tile_muls_f32(bc, bc, lr, n);
        tile_std::__tile_pipe_barrier();
        // param -= update
        tile_std::__tile_v_sub_f32(bc, bp, bc, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(param, bc, n);
        tile_std::__tile_buf_store_f32(cache, bt, n);
    }
}

/// Adam update (simplified):
///   m = beta1*m + (1-beta1)*grad
///   v = beta2*v + (1-beta2)*grad^2
///   param -= lr * m / (sqrt(v) + eps)
/// Maps to optimizer/adam.py
#[tile_std::tile_kernel]
pub fn adam_update(
    param: *mut f32, grad: *const f32,
    m_state: *mut f32, v_state: *mut f32,
    config: *const f32, len: *const u32
) {
    unsafe {
        let n = *len;
        let lr = *config;
        let beta1 = *config.wrapping_add(1);
        let beta2 = *config.wrapping_add(2);
        let eps = 1e-8f32;

        let bp = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        let bm = tile_std::__tile_buf_alloc(n);
        let bv = tile_std::__tile_buf_alloc(n);
        let bt = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(bp, param as *const f32, n);
        tile_std::__tile_buf_load_f32(bg, grad, n);
        tile_std::__tile_buf_load_f32(bm, m_state as *const f32, n);
        tile_std::__tile_buf_load_f32(bv, v_state as *const f32, n);
        tile_std::__tile_pipe_barrier();

        // m = beta1 * m
        tile_std::__tile_muls_f32(bm, bm, beta1, n);
        // bt = (1-beta1) * grad
        tile_std::__tile_muls_f32(bt, bg, 1.0f32 - beta1, n);
        tile_std::__tile_pipe_barrier();
        // m = beta1*m + (1-beta1)*grad → bt = new_m
        tile_std::__tile_v_add_f32(bt, bm, bt, n);
        tile_std::__tile_pipe_barrier();
        // bt now = new_m, save for later store

        // bm = grad^2 (reuse bm as temp, we saved new_m in bt)
        tile_std::__tile_v_mul_f32(bm, bg, bg, n);
        tile_std::__tile_pipe_barrier();
        // bm = (1-beta2) * grad^2
        tile_std::__tile_muls_f32(bm, bm, 1.0f32 - beta2, n);
        // v = beta2 * v
        tile_std::__tile_muls_f32(bv, bv, beta2, n);
        tile_std::__tile_pipe_barrier();
        // v = beta2*v + (1-beta2)*grad^2 → bm = new_v
        tile_std::__tile_v_add_f32(bm, bv, bm, n);
        tile_std::__tile_pipe_barrier();
        // bm = new_v, bt = new_m

        // bg = sqrt(v) + eps (reuse bg as temp)
        tile_std::__tile_v_sqrt_f32(bg, bm, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(bg, bg, eps, n);
        tile_std::__tile_pipe_barrier();
        // bg = m / (sqrt(v) + eps)
        tile_std::__tile_v_div_f32(bg, bt, bg, n);
        tile_std::__tile_pipe_barrier();
        // bg = lr * m / (sqrt(v) + eps)
        tile_std::__tile_muls_f32(bg, bg, lr, n);
        tile_std::__tile_pipe_barrier();
        // param -= update
        tile_std::__tile_v_sub_f32(bg, bp, bg, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(param, bg, n);
        tile_std::__tile_buf_store_f32(m_state, bt, n);
        tile_std::__tile_buf_store_f32(v_state, bm, n);
    }
}
