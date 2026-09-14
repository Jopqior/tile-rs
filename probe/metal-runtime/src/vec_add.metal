// Custom MSL compute kernel for the runner probe.
// Same source feeds both paths: runtime compilation via the `metal` crate
// (Device::new_library_with_source) and offline `xcrun metal -c` + `xcrun metallib`.
#include <metal_stdlib>
using namespace metal;

// f32 vector add. Expected output is computed independently on the CPU (Rust f32)
// and compared element-by-element after the GPU readback.
kernel void vec_add(device const float* a  [[buffer(0)]],
                    device const float* b  [[buffer(1)]],
                    device float* out      [[buffer(2)]],
                    uint id                [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}
