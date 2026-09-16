//! CPU reference implementations of broadcast/elementwise ext kernels.
//! Mirrors: broadcast_ext_kernel.rs

/// Where: out[i] = if mask[i] { x[i] } else { y[i] }
pub fn where_broadcast(x: &[f32], y: &[f32], mask: &[bool], output: &mut [f32]) {
    for i in 0..x.len() {
        output[i] = if mask[i] { x[i] } else { y[i] };
    }
}

/// Logical AND: out[i] = if a[i] != 0 && b[i] != 0 { 1.0 } else { 0.0 }
pub fn logic_and_broadcast(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..a.len() {
        output[i] = if a[i] != 0.0 && b[i] != 0.0 { 1.0 } else { 0.0 };
    }
}

/// Power: out[i] = base[i] ^ exp[i]
pub fn power_broadcast(base: &[f32], exp: &[f32], output: &mut [f32]) {
    for i in 0..base.len() {
        output[i] = base[i].powf(exp[i]);
    }
}
