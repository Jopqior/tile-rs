//! Metal runtime compile, dispatch, sync, and readback.
//!
//! On macOS this uses the `metal` crate (same path as the #9 probe). On other
//! OS the stage is unverified: there is no Metal device.

use crate::cases::Case;

pub struct GpuReadback {
    pub device_name: String,
    pub device_low_power: bool,
    pub device_headless: bool,
    pub threadgroup: u64,
    pub groups: u64,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub out: Vec<f32>,
}

pub enum GpuError {
    Unverified(String),
    Fail(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Unverified(s) | GpuError::Fail(s) => write!(f, "{s}"),
        }
    }
}

pub fn describe_host() -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("OS={}", std::env::consts::OS));
    lines.push(format!("ARCH={}", std::env::consts::ARCH));
    if let Ok(v) = std::process::Command::new("uname").arg("-a").output() {
        lines.push(format!("UNAME={}", String::from_utf8_lossy(&v.stdout).trim()));
    }
    #[cfg(target_os = "macos")]
    if let Ok(v) = std::process::Command::new("sw_vers").output() {
        for line in String::from_utf8_lossy(&v.stdout).lines() {
            lines.push(format!("SW_VERS {line}"));
        }
    }
    if let Ok(v) = std::env::var("ImageOS") {
        lines.push(format!("ImageOS={v}"));
    }
    if let Ok(v) = std::env::var("RUNNER_IMAGE_VERSION") {
        lines.push(format!("RUNNER_IMAGE_VERSION={v}"));
    }
    if let Ok(v) = std::env::var("RUNNER_OS") {
        lines.push(format!("RUNNER_OS={v}"));
    }
    if let Ok(v) = std::env::var("RUNNER_ARCH") {
        lines.push(format!("RUNNER_ARCH={v}"));
    }
    if let Ok(v) = std::env::var("GITHUB_RUN_ATTEMPT") {
        lines.push(format!("GITHUB_RUN_ATTEMPT={v}"));
    }
    lines
}

#[cfg(not(target_os = "macos"))]
pub fn run(_case: &Case, _msl: &str) -> Result<GpuReadback, GpuError> {
    Err(GpuError::Unverified(
        "Metal runtime is only available on macOS; this host cannot execute the kernel".into(),
    ))
}

#[cfg(target_os = "macos")]
pub fn run(case: &Case, msl: &str) -> Result<GpuReadback, GpuError> {
    macos::run(case, msl)
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use metal::*;
    use std::mem::size_of;

    fn fill_shared(device: &Device, data: &[f32]) -> Buffer {
        let len = (data.len() * size_of::<f32>()) as u64;
        let buf = device.new_buffer(len, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
        }
        buf
    }

    fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        (0..n).map(|i| unsafe { *ptr.add(i) }).collect()
    }

    pub fn run(case: &Case, msl: &str) -> Result<GpuReadback, GpuError> {
        let devices = Device::all();
        println!("METAL_DEVICE_COUNT={}", devices.len());
        for d in &devices {
            println!(
                "METAL_DEVICE name={:?} registry_id={} low_power={} headless={}",
                d.name(),
                d.registry_id(),
                d.is_low_power(),
                d.is_headless()
            );
        }
        let Some(device) = Device::system_default() else {
            return Err(GpuError::Unverified(
                "no system default Metal device".into(),
            ));
        };
        let device_name = device.name().to_string();
        println!(
            "METAL_DEVICE_SELECTED name={:?} low_power={} headless={}",
            device_name,
            device.is_low_power(),
            device.is_headless()
        );
        if device_name.contains("Paravirtual") {
            println!("METAL_DEVICE_NOTE=Apple Paravirtual device; not physical Apple GPU coverage");
        }

        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        println!("COMPILE_FAST_MATH=false");
        let library = device
            .new_library_with_source(msl, &opts)
            .map_err(|e| GpuError::Fail(format!("MSL compile failed: {e:?}")))?;
        let kernel = library
            .get_function(case.kernel_name, None)
            .map_err(|e| GpuError::Fail(format!("kernel {} not found: {e:?}", case.kernel_name)))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| GpuError::Fail(format!("pipeline failed: {e:?}")))?;

        let n = case.a.len() as u64;
        let max_tg = pipeline.max_total_threads_per_threadgroup() as u64;
        let max_tg = max_tg.max(1);
        // Instantiate from this pipeline; do not hard-code 32/256 as a contract.
        let threadgroup = max_tg.min(n.max(1));
        let groups = (n + threadgroup - 1) / threadgroup;
        println!(
            "DISPATCH n={n} threadgroup={threadgroup} groups={groups} pipeline_max_total_threads_per_threadgroup={max_tg}"
        );

        let a_buf = fill_shared(&device, case.a);
        let b_buf = fill_shared(&device, case.b);
        let out_init: Vec<f32> = vec![crate::cases::OUTPUT_SENTINEL; case.a.len() + case.preserve_pad];
        let out_buf = fill_shared(&device, &out_init);
        let n_u32 = n as u32;
        let n_buf = fill_shared_u32(&device, n_u32);

        let queue = device.new_command_queue();
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_buffer(3, Some(&n_buf), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(threadgroup, 1, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Ok(GpuReadback {
            device_name,
            device_low_power: device.is_low_power(),
            device_headless: device.is_headless(),
            threadgroup,
            groups,
            a: read_f32(&a_buf, case.a.len()),
            b: read_f32(&b_buf, case.b.len()),
            out: read_f32(&out_buf, case.a.len() + case.preserve_pad),
        })
    }

    fn fill_shared_u32(device: &Device, value: u32) -> Buffer {
        let buf = device.new_buffer(size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);
        unsafe {
            *(buf.contents() as *mut u32) = value;
        }
        buf
    }
}
