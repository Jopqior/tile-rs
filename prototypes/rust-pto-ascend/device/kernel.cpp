// THROWAWAY: launch wrapper only. generated.cpp must come from the Rust job.
#include <cstdint>
#include <cstddef>
#include <pto/pto-inst.hpp>
#include "generated.cpp"

__global__ AICORE void add_kernel(__gm__ float* a, __gm__ float* b, __gm__ float* out) {
    add_generated(a, b, out);
}
extern "C" void launch_add(float* a, float* b, float* out, void* stream) {
    add_kernel<<<1, nullptr, stream>>>(a, b, out);
}
