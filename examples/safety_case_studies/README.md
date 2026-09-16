# Memory Safety Case Studies: AscendC C++ vs tile-rs Rust

This directory contains paired examples where each AscendC C++ kernel contains a
real, exploitable memory safety vulnerability that the equivalent Rust tile-rs
kernel structurally prevents.

These are not contrived toy examples. Each vulnerability class is a real pattern
that occurs in production AscendC C++ kernel development. The Rust versions show
how the tile-rs API design makes each class of bug either impossible or requires
explicit `unsafe` opt-in.

## Summary

| Case | Vulnerability | C++ Root Cause | Rust Prevention |
|------|--------------|----------------|-----------------|
| 1 | Type Confusion | `GM_ADDR` erases all type info at kernel entry | Function signature encodes element type |
| 2 | Buffer Overflow | `GetValue(i)`/`SetValue(i,v)` have no bounds check | Buffer-ID API with explicit count parameter |
| 3 | Use-After-Free | `FreeTensor()` then access via stale `LocalTensor` | No manual free in API; buffer IDs are integers |
| 4 | Missing Sync | Forgetting `pipe_barrier()` between DMA and compute | `kernel_ops` composites include barriers internally |
| 5 | Double Free | `FreeTensor()` called twice on same tensor | No free operation exists in the API |
| 6 | Integer Overflow | Silent `u32` wrap in multi-block offset calculation | `wrapping_mul` makes overflow semantics explicit |

## How to Read Each Case

Each subdirectory contains two files:

- **`vulnerable.cpp`** — A complete AscendC C++ kernel with a realistic bug.
  The exact vulnerability is marked with a `// BUG:` comment.
- **`safe.rs`** — The equivalent tile-rs Rust kernel that structurally
  prevents the same class of bug. Comments explain which language or API
  mechanism provides the safety guarantee.

The C++ files follow standard AscendC conventions: class-based kernel with
`Init`/`Process`/`CopyIn`/`Compute`/`CopyOut` methods, `TPipe` for buffer
management, and `extern "C" __global__ __aicore__` entry points.

The Rust files follow the tile-rs compiletest pattern: `#![no_core]` crate
with `#[tile_std::aiv_kernel]` attribute, buffer-ID based allocation, and
explicit `__tile_pipe_barrier()` calls.

## Case Details

### Case 1: Type Confusion via GM_ADDR Type Erasure

AscendC kernel entry points receive `GM_ADDR` (`uint8_t*`). The kernel manually
casts to a typed `GlobalTensor<T>`. If the host passes f16 data but the kernel
casts to `float*`, elements are reinterpreted with wrong size — 2x the data is
read, each value containing garbage bit patterns. In Rust, the function signature
encodes the pointer type; passing `*const u16` where `*const f32` is expected is
a compile error.

### Case 2: Buffer Overflow via Unchecked Tensor Index

AscendC's `GetValue(i)` and `SetValue(i, v)` perform no bounds checking. An
off-by-one loop bound or using the wrong length variable causes out-of-bounds
reads/writes on local SRAM — corrupting adjacent tensor data or crashing the NPU.
In Rust, vector operations take an explicit count parameter and operate on opaque
buffer IDs, not raw pointers with unchecked indexing.

### Case 3: Use-After-Free of LocalTensor

AscendC requires manual `FreeTensor()` calls. A freed tensor's `LocalTensor`
handle remains valid at the C++ type level, so subsequent `GetValue()` calls
compile and run — reading from SRAM that has been returned to the queue's free
pool and may be reallocated. In Rust, buffer IDs are plain integers with no
free operation; the kernel generator manages buffer lifetimes automatically.

### Case 4: Missing Synchronization Between Pipeline Stages

Ascend NPUs execute DMA, vector, and scalar pipelines concurrently. Forgetting
`pipe_barrier()` between a DMA load and a subsequent vector operation means the
vector unit reads stale or uninitialized data. This is the most common NPU bug.
In Rust, `kernel_ops` composites (sigmoid, softmax, etc.) include barriers
between every pipeline-crossing step.

### Case 5: Double-Free of Tensor Buffers

Calling `FreeTensor()` twice on the same `LocalTensor` corrupts the queue's
internal free list. The next `AllocTensor()` may return an overlapping buffer,
causing silent data corruption in subsequent tiles. In Rust, no free operation
exists — buffer IDs are just integers, and using the same ID twice is harmless.

### Case 6: Silent Integer Overflow in Multi-Block Offset

Multi-block kernels compute `offset = blockIdx * perBlockLen`. With u32
arithmetic and large tensors, this silently overflows, wrapping around to address
the wrong memory region. In Rust, `wrapping_mul` makes overflow semantics
explicit and intentional. Debug builds detect overflow via panic.
