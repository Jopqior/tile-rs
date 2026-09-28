use tile_kernel_builder::KernelBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=vec_add_op.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    tile_kernel_builder::add_ascend_link_args()?;
    let kernel = KernelBuilder::new("hello_world.cpp").build()?;

    println!(
        "cargo:rustc-link-search=native={}",
        kernel.parent().unwrap().display()
    );
    println!("cargo:rustc-link-lib=dylib=kernels");

    Ok(())
}
