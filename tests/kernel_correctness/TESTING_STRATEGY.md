# Kernel Correctness Testing Strategy

## Overview

Three-layer testing architecture for 100% kernel coverage without NPU hardware.

## Layer 1: Compilation Tests (existing)
- `cargo run -p tests --release` — all 367 kernels compile through MLIR backend
- Catches type errors, missing intrinsics, codegen ICEs

## Layer 2: CPU Golden-Value Tests
- `tests/kernel_correctness/` — regular Rust crate, no ascend_std dependency
- Extracts scalar-loop kernel logic as plain Rust functions
- Tests against hand-crafted + numpy/torch golden values
- Covers all 68 new kernels (conv, index, pooling, matmul, resize, etc.)

## Layer 3: Property-Based / Randomized Tests
- Pre-generated random test cases from PyTorch (golden JSON files)
- 50-100 random cases per kernel class
- Covers edge cases: padding boundaries, dilation, stride, empty inputs

## Running
```bash
# Generate golden values (requires Python + torch)
python3 tests/kernel_correctness/golden/generate.py

# Run all correctness tests
cargo test -p kernel_correctness --release

# Check kernel function sync between compiletest and CPU test files
bash scripts/check_kernel_sync.sh
```
