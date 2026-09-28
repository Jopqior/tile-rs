// pto-compat-a3.hpp — Compatibility shim for pto-isa a2a3 headers (dav-c220, Ascend910_9362..9392).
//
// Used when pto_arch=a3 (the a2a3 pto-isa path). The 910c dev server has Ascend910_9392 chips
// which are dav-c220 (Ascend910B), NOT original Ascend910 (dav-c100). The a2a3 pto-isa path
// is used because Ascend910_9362..9392 predates the Ascend910B2/B4 a5 series.
//
// When compiling with CANN 8.5.0 ccec + -DMEMORY_BASE targeting dav-c220-vec, several items
// in pto-isa headers may be absent depending on the CANN version and chip variant:
//
//   1. PIPE_FIX — fixpipe hardware not present on dav-c100 (__CCE_AICORE__=100).
//      On dav-c220 PIPE_FIX is already defined by ccec, so the #ifndef guard is a no-op.
//      Included for safety when also targeting real dav-c100 (original Ascend910).
//
//   2. bfloat16_t — not defined by some ccec versions. Cannot alias to `half` because
//      constants.hpp has separate template specializations for both.
//      Defined as a distinct stub struct (no-op on chips that already define it).
//
//   3. __biasbuf__ / __fbuf__ — CCE memory attributes rejected in __tf__ context on some
//      targets. Redirected to __cbuf__ / __ubuf__ (nearest equivalents).
//
//   4. st_atomic<T> — atomic store intrinsic used in TNotify.hpp. Stub provided for
//      headers-only parse (never called by softmax kernel).
//
// Include BEFORE pto-isa headers via: ccec -include pto-compat-a3.hpp
#pragma once

// 1. PIPE_FIX — not available on dav-c100 (__CCE_AICORE__ < 210).
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_ALL
#endif

// 2. bfloat16_t — not defined by dav-c100 ccec. On dav-c220+ ccec already typedef's it
//    as __bf16 in __clang_cce_types.h, so the stub must only appear on dav-c100.
//    Cannot alias to `half` because constants.hpp has separate template specializations.
//    Guard: __CCE_AICORE__ == 100 for dav-c100; dav-c220 has __CCE_AICORE__ >= 220.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 200
#ifndef __bfloat16_t_defined
#define __bfloat16_t_defined
struct bfloat16_t {
    unsigned short bits;
    bfloat16_t() = default;
    explicit bfloat16_t(float) : bits(0) {}
    operator float() const { return 0.0f; }
};
#endif
#endif

// 3. __biasbuf__ / __fbuf__ — these CCE memory attributes are defined as macros by
//    ccec's __clang_cce_defines.h (e.g. __biasbuf__ = __attribute__((cce_bias_table_buff))).
//    On dav-c100 the compiler rejects them in __tf__ function context with a semantic error.
//    They appear in TMov.hpp/TInsert.hpp matrix ops that are never called for softmax, but
//    the function bodies are parsed. Redefine them to available equivalents.
#undef __biasbuf__
#define __biasbuf__ __cbuf__
#undef __fbuf__
#define __fbuf__ __ubuf__

// 4. st_atomic<T> — not declared by dav-c100 ccec intrinsics. Used in TNotify.hpp
//    (inter-core atomic store). Must be an [aicore] function to be callable from aicore context.
//    The `[aicore]` C++17 attribute is the ccec way to mark aicore device functions.
//    TNotify is never instantiated in softmax; this stub just allows the header to parse.
#ifndef __st_atomic_stub_defined
#define __st_atomic_stub_defined
template <typename T>
[aicore] static inline void st_atomic(T value, __gm__ T *addr) { *addr = value; }
#endif
