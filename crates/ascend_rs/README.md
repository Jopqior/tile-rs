# ascend_rs — Huawei Ascend NPU backend for tile-rs

This is the **Ascend NPU backend** for [tile-rs](../../README.md): the host runtime
and CANN/ACL integration behind the `cpp` and `pto` codegen targets. tile-rs is a
multi-backend framework — Ascend is one of its 15 targets, and one of three
validated end-to-end on silicon. Everything below is Ascend-specific.

## Crates

| Crate | Purpose |
|-------|---------|
| `ascend_rs` | Host runtime: ACL, device/context/stream/memory, kernel launch, model inference, DVPP, profiling, HCCL |
| `ascend_sys` | Raw FFI bindings generated via `bindgen` from CANN C++ headers |
| `ascend_rs_blas` | BLAS operations on Ascend NPU |

The kernel-side device runtime is `tile_std` (`#[aiv_kernel]` lowers to AscendC via
`ACLRS_CODEGEN_PATH=cpp`/`pto`); `tile_kernel_builder` drives the CANN/bisheng
toolchain.

## Quick Start

### One-liner install

```bash
curl -fsSL https://raw.githubusercontent.com/yijunyu/tile-rs/main/scripts/install.sh | bash
```

This auto-detects your environment and does the right thing:
- **NPU hardware detected** → runs natively on device
- **CANN toolkit only** → runs in simulator mode
- **Neither (Docker available)** → pulls `ghcr.io/trusted-programming/ascend-ci:latest` and runs in simulator

### Manual setup

**Prerequisites**

- **Ascend Toolkit** (CANN 8.x/9.x) installed, **or** Docker for simulator mode
- **Rust nightly** (`nightly-2025-08-04`, pinned in `rust-toolchain.toml`)

**With CANN installed**

```bash
# Set CANN path (auto-detected from defaults if unset)
export ACLRS_CANN_PATH="/usr/local/Ascend/ascend-toolkit/latest"

# Source CANN environment
source ${ACLRS_CANN_PATH}/bin/setenv.bash

# Set run mode: npu (real hardware) or sim (simulator)
export ACLRS_RUN_MODE=npu
```

**Without CANN (Docker simulator)**

```bash
git clone https://github.com/yijunyu/tile-rs.git && cd tile-rs
docker run --rm \
  -v "$PWD:/workspace" -v "$HOME/.cargo:/root/.cargo" -v "$HOME/.rustup:/root/.rustup" \
  -w /workspace/examples/acl_hello_world \
  -e ACLRS_RUN_MODE=sim -e ACLRS_SOC_VERSION=Ascend310P1 \
  -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
  ghcr.io/trusted-programming/ascend-ci:latest bash -c '
    export PATH="/root/.cargo/bin:$PATH"
    SIM=/usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/simulator/Ascend310P1/lib
    export LD_LIBRARY_PATH="$SIM:$(rustc --print sysroot)/lib:$LD_LIBRARY_PATH"
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    cargo run --release
  '
```

### Writing a Rust NPU kernel

```rust
#![feature(no_core)]
#![no_std]
#![no_core]

#[tile_std::aiv_kernel]
pub unsafe fn vec_add(x: *const f32, y: *const f32, z: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf_x = tile_std::__tile_buf_alloc(n);
        let buf_y = tile_std::__tile_buf_alloc(n);
        let buf_z = tile_std::__tile_buf_alloc(n);

        tile_std::__tile_buf_load_f32(buf_x, x, n);
        tile_std::__tile_buf_load_f32(buf_y, y, n);
        tile_std::__tile_pipe_barrier();

        tile_std::__tile_v_add_f32(buf_z, buf_x, buf_y, n);

        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(z, buf_z, n);
    }
}
```

### Build script (`build.rs`)

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tile_kernel_builder::add_ascend_link_args()?;
    tile_kernel_builder::KernelBuilder::new("kernels").build()?;
    Ok(())
}
```

### Cargo.toml

```toml
[dependencies]
ascend_rs = { path = "path/to/crates/ascend_rs" }

[build-dependencies]
tile_kernel_builder = { path = "path/to/crates/tile_kernel_builder" }
```

### Run

```bash
cargo run
```

## Build commands

```bash
# Build workspace (excluding hardware-dependent examples)
cargo build --release --workspace \
  --exclude acl_hello_world --exclude acl_rs_vec_add \
  --exclude kernels --exclude acl_vec_add \
  --exclude common --exclude gemm

# Check formatting
cargo fmt --check --all

# Run compiletest UI tests (requires MLIR codegen backend)
cargo test -p compiletest
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `ACLRS_CODEGEN_PATH` | Selects the target backend — `cpp`/`pto` (Ascend), `metal`, `cuda`, `vulkan`, `nki`, `aie`, … Defaults to the Ascend C++ path. |
| `ASCEND_LOG` | Tracing filter for the codegen backend (e.g. `trace`) |
| `ACLRS_CANN_PATH` | Path to the Ascend Toolkit (auto-detected if unset) |
| `ACLRS_RUN_MODE` | `npu` (real hardware) or `sim` (simulator) |
| `ACLRS_SOC_VERSION` | Chip version, e.g. `Ascend310P1` (simulator only) |
| `CAMODEL_LOG_PATH` | Simulator log folder (simulator only) |

## Kernel coverage

The Ascend NPU backend ships **502 Rust kernels** with complete 1:1 coverage of all 300 MultiKernelBench reference kernels across 17 categories:

| Category | Kernels | Examples |
|----------|---------|---------|
| Activation | 16 | relu, sigmoid, gelu, tanh, softmax, elu, swish, mish, hardsigmoid, ... |
| Architecture | 41 | MLP, ResNet residual, ViT MLP, LSTM/GRU cells, Mamba SSM, ... |
| Attention | 15 | scaled dot-product, causal, cross, multi-query, KV-cached, sparse, ... |
| Broadcast | 8 | add_bias, elementwise mul/div/sub/max/min, clamp, square |
| Convolution | 34 | standard conv2d, depthwise conv2d, transposed conv2d variants |
| Fuse | 86 | matmul+activation chains, norm+activation, gemm+misc, multi-op fusion |
| Index | 12 | gather, scatter, scatter_add, index_select, embedding, masked_fill, ... |
| Loss | 6 | MSE, Huber, hinge, cosine similarity, cross-entropy, KL divergence |
| Math | 5 | cumsum (4 variants), cumprod |
| Matmul | 17 | standard, batched, symmetric, GEMM, diagonal-scale, outer product, ... |
| Normalization | 9 | layernorm, rmsnorm, batch/group/instance norm, L1/L2/Frobenius norm |
| Optimizer | 6 | SGD, SGD+momentum, Adagrad, RMSprop, Adam, extended |
| Pooling | 6 | global avg/max/min pool, fused pool+sigmoid, LP pool |
| Reduce | 5 | max, min, sum, mean, product |
| Resize | 5 | nearest, lerp, bicubic weight, weighted sum, trilinear |
| Tiled | 16 | 256-element tiled variants of activations and ops |
| Multi-block | 16 | AICore block-parallel variants of activations and ops |

In addition to 486 compiletest kernels, there are **16 deployable kernel examples** with host code and **522 tests passing NPU correctness verification** on real Ascend 910B3 hardware (verified against CPU reference, 0 failures).

Count kernels: `bash scripts/count_kernels.sh` | Full inventory: `bash scripts/generate_kernel_appendix.sh`

## Memory safety case studies

The [`examples/safety_case_studies/`](../../examples/safety_case_studies/) directory contains 6 paired examples demonstrating tile-rs's memory safety advantages over AscendC C++. Each case pairs a C++ kernel containing a real, exploitable vulnerability with a Rust kernel that structurally prevents the same class of bug:

| Case | Vulnerability | C++ Root Cause | Rust Prevention |
|------|--------------|----------------|-----------------|
| Type Confusion | `GM_ADDR` erases all type info | Typed `*const f32` in function signature |
| Buffer Overflow | `GetValue(i)` unchecked indexing | Buffer-ID API with explicit count |
| Use-After-Free | `FreeTensor` then stale access | No manual free in API |
| Missing Sync | Forgotten `pipe_barrier()` | `kernel_ops` composites include barriers |
| Double Free | `FreeTensor` called twice | No free operation exists |
| Integer Overflow | Silent `u32` wrap | `wrapping_mul` makes overflow explicit |

These are not contrived examples — each vulnerability class is a real pattern that occurs in production AscendC C++ kernel development.

## Testing

**Compiletest suite** — 486 UI tests in `tests/compiletest/ui/` validate the MLIR codegen backend across kernel operations (activations, matmul, attention, fused chains, tiling, multi-block, convolution, index/scatter) and language features.

**NPU correctness tests** — 522 tests in `tests/npu_correctness/` run on real Ascend 910B3 hardware and verify output against CPU reference with 0 failures (activations, normalization, loss, matmul, attention, convolution, pooling, resize, index/scatter, optimizer, fused multi-op chains).

## Troubleshooting

- **`aclInit failed with error code: 507008`** — Check Ascend driver/hardware configuration
- **`cannot open shared object file`** — Verify `LD_LIBRARY_PATH` or use `add_ascend_link_args()` in build script
- **`symbol lookup error`** — Library version mismatch; ensure all Ascend Toolkit libraries come from the same installation
- Use `cargo run -v` for verbose build output

## Acknowledgements

This work builds on **[aspect-rs](https://github.com/trusted-programming/aspect-rs)**,
the earlier project that pioneered the Rust-to-accelerator approach this backend
extends. We gratefully acknowledge **Kirill**, **Luca**, and **Vadim** for that
foundational work on aspect-rs.
