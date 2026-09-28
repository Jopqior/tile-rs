// =============================================================================
// Build Script: Compiles the NPU kernel at build time
// =============================================================================
//
// This build script runs before the host binary is compiled. It uses
// `KernelBuilder` to compile the Rust kernel crate into an NPU binary.
//
// The build flow:
//   1. KernelBuilder::new("kernels") detects a directory -> Rust kernel
//   2. Internally invokes `cargo build --lib --target=davinci-huawei-none`
//      with `-Zcodegen-backend=rustc_codegen_tile`
//   3. The MLIR codegen backend produces a `kernel.acl.o` artifact
//   4. The artifact is copied to `$OUT_DIR/kernel.o`
//   5. At runtime, `KernelLoader::new()` reads this file from `$OUT_DIR`

use tile_kernel_builder::KernelBuilder;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=build.rs");
    tile_kernel_builder::add_ascend_link_args()?;

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernel = out_path.join("kernel.o");
    KernelBuilder::new("kernels").copy_to(&kernel).build()?;
    Ok(())
}
