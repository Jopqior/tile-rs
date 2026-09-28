use tile_kernel_builder_config::Settings;

use std::io::Write;
use std::path::{Path, PathBuf};

use super::BuilderError;

pub(super) fn build(kernel_path: impl AsRef<Path>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    assert!(kernel_path.as_ref().is_file());

    let settings = Settings::new()?;
    let cann = settings.cann();
    // FIXME: checks for existance and path inference
    let ascend_kernel_cmake = format!("{}/tools/tikcpp/ascendc_kernel_cmake/", cann.display());
    let run_mode = settings.run_mode();
    let soc_version = run_mode.soc_version();

    let kernel_file = kernel_path
        .as_ref()
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(BuilderError::PathDoesntExist(
            kernel_path.as_ref().to_owned(),
        ))?;

    // create cmake project for kernel:
    // ${OUT_DIR}/kernels:
    //     - CMakeLists.txt
    //     - kernel_file
    let cmake_project_path: PathBuf = std::env::var("OUT_DIR")?.into();
    let cmake_project_path = cmake_project_path.join("kernels");
    if cmake_project_path.exists() {
        std::fs::remove_dir_all(&cmake_project_path)?;
    }
    std::fs::create_dir(&cmake_project_path)?;
    let cmake = format!(
        "
            cmake_minimum_required(VERSION 3.16.9)
            project(Ascend_kernel)

            include({}/ascendc.cmake)

            ascendc_library(kernels SHARED {})
        ",
        ascend_kernel_cmake, kernel_file,
    );
    let cmake_path = cmake_project_path.join("CMakeLists.txt");
    let mut cmake_file = std::fs::File::create(&cmake_path)?;
    cmake_file.write_all(cmake.as_bytes())?;
    let kernel_output_path = cmake_project_path.join(kernel_file);
    std::fs::copy(kernel_path, &kernel_output_path)?;

    // build kernel
    let dst = cmake::Config::new(cmake_project_path)
        .define("SOC_VERSION", soc_version)
        .define("RUN_MODE", run_mode.to_string())
        .define("ASCEND_CANN_PACKAGE_PATH", cann)
        .build();

    if !dst.exists() {
        return Err(Box::new(BuilderError::BuildFailed));
    }

    Ok(dst.join("build/lib/libkernels.so"))
}
