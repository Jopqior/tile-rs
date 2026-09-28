// build-pass

// Extended broadcast/elementwise operation kernels.
// Maps to MultiKernelBench/reference/broadcast/ category (remaining ops).

#![feature(no_core)]

#![no_std]
#![no_core]

/// Where broadcast: dst[i] = if mask[i] != 0 { x[i] } else { y[i] }
/// Maps to broadcast/where_broadcast.py
#[tile_std::tile_kernel]
pub fn where_broadcast(
    x: *const f32, y: *const f32, mask: *const u32, output: *mut f32, len: *const u32,
) {
    unsafe {
        let n = *len;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let m = *mask.wrapping_add(i as usize);
            if m != 0 {
                *output.wrapping_add(i as usize) = *x.wrapping_add(i as usize);
            } else {
                *output.wrapping_add(i as usize) = *y.wrapping_add(i as usize);
            }
            i = i + 1;
        }
    }
}

/// Logical AND broadcast: dst[i] = (a[i] != 0) & (b[i] != 0) ? 1.0 : 0.0
/// Maps to broadcast/logic_and_broadcast.py
#[tile_std::tile_kernel]
pub fn logic_and_broadcast(
    a: *const f32, b: *const f32, output: *mut f32, len: *const u32,
) {
    unsafe {
        let n = *len;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let va = *a.wrapping_add(i as usize);
            let vb = *b.wrapping_add(i as usize);
            if va != 0.0f32 && vb != 0.0f32 {
                *output.wrapping_add(i as usize) = 1.0f32;
            } else {
                *output.wrapping_add(i as usize) = 0.0f32;
            }
            i = i + 1;
        }
    }
}

/// Power broadcast: dst[i] = base[i] ^ exp[i] = exp(exp[i] * ln(base[i]))
/// Maps to broadcast/power_broadcast.py (general power, not just square)
#[tile_std::tile_kernel]
pub fn power_broadcast(
    base: *const f32, exp_buf: *const f32, output: *mut f32, len: *const u32,
) {
    unsafe {
        let n = *len;
        let mut i = 0u32;
        loop {
            if i >= n { break; }
            let b = *base.wrapping_add(i as usize);
            let e = *exp_buf.wrapping_add(i as usize);
            // pow(b, e) = exp(e * ln(b))
            let ln_b = tile_std::core::builtins::logf(b);
            let result = tile_std::core::builtins::expf(e * ln_b);
            *output.wrapping_add(i as usize) = result;
            i = i + 1;
        }
    }
}
