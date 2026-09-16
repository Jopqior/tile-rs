use anyhow::{Result, anyhow};
use std::fs;
use std::path::Path;
use tile_kernel_builder_config::Settings;

pub fn build_operator(descr: impl AsRef<Path>, out_dir: impl AsRef<Path>) -> Result<()> {
    let settings = Settings::new()?;
    let run_mode = settings.run_mode();
    let soc_version = run_mode.soc_version();

    let out_dir = out_dir.as_ref();
    if !out_dir.exists() {
        fs::create_dir_all(out_dir)?;
    }

    let output = std::process::Command::new("atc")
        .arg(format!("--singleop={}", descr.as_ref().display()))
        .arg(format!("--soc_version={}", soc_version))
        .arg(format!("--output={}", out_dir.display()))
        .output()?;

    if !output.status.success() {
        return Err(anyhow!(String::from_utf8(output.stdout).unwrap()));
    }

    Ok(())
}
