use tile_kernel_builder::KernelBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=softmax_kernels.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    tile_kernel_builder::add_ascend_link_args()?;
    let kernel = KernelBuilder::new("softmax_kernels.cpp").build()?;

    let kernel_dir = kernel.parent().unwrap();
    println!("cargo:rustc-link-search=native={}", kernel_dir.display());
    println!("cargo:rustc-link-lib=dylib=kernels");
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", kernel_dir.display());

    Ok(())
}
