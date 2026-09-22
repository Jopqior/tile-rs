// THROWAWAY: real A2 vector device kernel, never CPU-SIM.
#include <cstdint>
#include <cstddef>
#include <pto/pto-inst.hpp>
#ifdef GENERATED
#include "generated.cpp" // Unmodified PTOAS output; only the launch wrapper is ours.
#endif
using namespace pto;
__global__ AICORE void add_kernel(__gm__ float* a, __gm__ float* b, __gm__ float* out) {
#ifdef GENERATED
    add_generated(a, b, out);
#else
    constexpr int n = 256;
    using Global = GlobalTensor<float, Shape<1, 1, 1, 1, n>, Stride<n, n, n, n, 1>>;
    using Vector = Tile<TileType::Vec, float, 1, n, BLayout::RowMajor, 1, n>;
    Vector x, y, z;
    TASSIGN(x, 0);
    TASSIGN(y, n * sizeof(float));
    TASSIGN(z, 2 * n * sizeof(float));
    Global ga(a), gb(b), gc(out);
    TLOAD(x, ga);
    TLOAD(y, gb);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(z, x, y);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gc, z);
    pipe_barrier(PIPE_ALL);
#endif
}
extern "C" void launch_add(float* a, float* b, float* out, void* stream) {
    add_kernel<<<1, nullptr, stream>>>(a, b, out);
}
