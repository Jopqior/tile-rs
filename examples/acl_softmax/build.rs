// =============================================================================
// Build Script: Compiles the NPU softmax kernel at build time
// =============================================================================

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
