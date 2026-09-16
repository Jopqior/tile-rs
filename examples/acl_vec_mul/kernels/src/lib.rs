// =============================================================================
// NPU Kernel: Element-wise Vector Multiplication
// =============================================================================
//
// This file defines a kernel that runs on the Ascend NPU (Neural Processing Unit).
//
// Compilation pipeline:
//   Rust source
//     -> rustc with `-Zcodegen-backend=rustc_codegen_tile` (produces MLIR)
//     -> MLIR lowering to Ascend NPU IR
//     -> kernel.acl.o (ELF binary for NPU)
//
// The kernel uses `#![no_core]` because the NPU has no operating system or
// standard library. Instead, `tile_std` provides a minimal reimplementation
// of Rust's core primitives (Copy, Clone, Add, Mul, etc.) that the codegen
// backend understands.

#![feature(no_core)]
#![no_std]
#![no_core]

/// Element-wise multiplication: z[i] = x[i] * y[i]
///
/// The `#[tile_std::tile_kernel]` attribute marks this function as an
/// AIV (Ascend Instruction Vector) kernel entry point. It expands to:
///   - `#[unsafe(no_mangle)]` so the host can look up the symbol by name
///   - `#[ascend::aiv_kernel]` which the MLIR codegen backend recognizes
///
/// Parameters are raw pointers to device memory buffers allocated by the host.
/// The kernel is launched with `block_dim` parallel blocks; each block
/// processes a disjoint slice of the data.
#[tile_std::tile_kernel]
pub fn mul(x: *const u16, y: *const u16, z: *mut u16) {
    unsafe {
        // Total elements = 16. Divide work evenly across blocks.
        let block_size = 16usize / tile_std::get_block_num();
        let start = tile_std::get_block_idx() * block_size;
        let mut i = start;
        loop {
            *z.wrapping_add(i) = *x.wrapping_add(i) * *y.wrapping_add(i);

            i = i + 1;
            if i == block_size + start {
                break;
            }
        }
    }
}
