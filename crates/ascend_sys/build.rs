use std::env;
use std::path::PathBuf;
use tile_kernel_builder_config::Settings;

fn link_native_libraries(settings: &Settings) {
    if let Some(cammodel_path) = settings.cann_sim() {
        println!("cargo:rustc-link-search=native={}", cammodel_path.display());
        println!("cargo:rustc-link-lib=dylib=runtime_camodel");
    }

    println!(
        "cargo:rustc-link-search=native={}",
        settings.cann_ascend_lib().display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        settings.cann_devlib().display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        settings.cann_ops_lib().display()
    );

    println!("cargo:rustc-link-lib=dylib=ascendcl");

    if cfg!(feature = "dvpp") {
        println!("cargo:rustc-link-lib=dylib=acl_dvpp");
    }
    if cfg!(feature = "blas") {
        println!("cargo:rustc-link-lib=dylib=acl_cblas");
    }
    if cfg!(feature = "hccl") {
        println!("cargo:rustc-link-lib=dylib=hccl");
    }
    if cfg!(feature = "profiler") {
        println!("cargo:rustc-link-lib=dylib=msprofiler");
    }

    // Prints all dynamic loading as it happens
    // println!("cargo:rustc-env=LD_DEBUG=all");

    // Prints shared objects loaded at launch for debugging,
    // not useful for post-launch dlopen() loads
    // println!("cargo:rustc-env=LD_TRACE_LOADED_OBJECTS=");
}

fn gen_core_bindings(settings: &Settings) -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=core_wrapper.h");

    let builder = bindgen::Builder::default()
        .header("core_wrapper.h")
        .clang_args(
            settings
                .cann_include_paths()
                .iter()
                .map(|path| format!("-I{}", path.display())),
        )
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    let bindings = builder.generate()?;

    let out_path = PathBuf::from(env::var("OUT_DIR")?);
    bindings.write_to_file(out_path.join("core_bindings.rs"))?;

    Ok(())
}

fn gen_blas_bindings(settings: &Settings) -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=blas_wrapper.h");

    let builder = bindgen::Builder::default()
        .header("blas_wrapper.h")
        .clang_args(
            settings
                .cann_include_paths()
                .iter()
                .map(|path| format!("-I{}", path.display())),
        )
        .clang_arg(settings.cann_ops_include().to_str().unwrap())
        .allowlist_function("^aclblas.*")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    let bindings = builder.generate()?;

    let out_path = PathBuf::from(env::var("OUT_DIR")?);
    bindings.write_to_file(out_path.join("blas_bindings.rs"))?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Settings::new()?;
    link_native_libraries(&config);
    gen_core_bindings(&config)?;
    if cfg!(feature = "blas") {
        gen_blas_bindings(&config)?;
    }
    Ok(())
}
