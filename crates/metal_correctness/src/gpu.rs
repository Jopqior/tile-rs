//! Metal runtime compile, dispatch, sync, and readback.
//!
//! On macOS this uses the `metal` crate (same path as the #9 probe). On other
//! OS the stage is unverified: there is no Metal device.
//!
//! The device entry point discovers the SIMD width and maximum threadgroup
//! size from a probe compiled through the same current-source emitter, and the
//! case sizes are instantiated from that. Dispatch uses the discovered
//! threadgroup size; it is never a hard-coded grid.

use crate::cases::{Case, DeviceCaps};

pub struct GpuReadback {
    pub device_name: String,
    pub device_low_power: bool,
    pub device_headless: bool,
    pub threadgroup: u64,
    pub groups: u64,
    pub pipeline_threadgroup_max: u64,
    pub thread_execution_width: u64,
    pub inputs: Vec<Vec<f32>>,
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
pub fn discover_caps(_probe_msl: &str, _probe_kernel: &str) -> Result<DeviceCaps, GpuError> {
    Err(GpuError::Unverified(
        "Metal runtime is only available on macOS; cannot discover device caps".into(),
    ))
}

#[cfg(target_os = "macos")]
pub fn discover_caps(probe_msl: &str, probe_kernel: &str) -> Result<DeviceCaps, GpuError> {
    macos::discover_caps(probe_msl, probe_kernel)
}

#[cfg(not(target_os = "macos"))]
pub fn run(_case: &Case, _msl: &str, _caps: &DeviceCaps) -> Result<GpuReadback, GpuError> {
    Err(GpuError::Unverified(
        "Metal runtime is only available on macOS; this host cannot execute the kernel".into(),
    ))
}

#[cfg(target_os = "macos")]
pub fn run(case: &Case, msl: &str, caps: &DeviceCaps) -> Result<GpuReadback, GpuError> {
    macos::run(case, msl, caps)
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

    fn fill_shared_u32(device: &Device, value: u32) -> Buffer {
        let buf = device.new_buffer(size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);
        unsafe {
            *(buf.contents() as *mut u32) = value;
        }
        buf
    }

    fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        (0..n).map(|i| unsafe { *ptr.add(i) }).collect()
    }

    fn select_device() -> Result<Device, GpuError> {
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
        let device = Device::system_default()
            .ok_or_else(|| GpuError::Unverified("no system default Metal device".into()))?;
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
        Ok(device)
    }

    fn compile(device: &Device, msl: &str, kernel_name: &str) -> Result<ComputePipelineState, GpuError> {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(msl, &opts)
            .map_err(|e| GpuError::Fail(format!("MSL compile failed: {e:?}")))?;
        let kernel = library
            .get_function(kernel_name, None)
            .map_err(|e| GpuError::Fail(format!("kernel {kernel_name} not found: {e:?}")))?;
        device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| GpuError::Fail(format!("pipeline failed: {e:?}")))
    }

    /// Discover SIMD width and max threadgroup size from a probe emitted by the
    /// current source emitter, not from a hard-coded constant.
    pub fn discover_caps(probe_msl: &str, probe_kernel: &str) -> Result<DeviceCaps, GpuError> {
        let device = select_device()?;
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        println!("COMPILE_FAST_MATH=false");
        let pipeline = compile(&device, probe_msl, probe_kernel)?;
        let simd_width = pipeline.thread_execution_width().max(1) as usize;
        let mtpt = device.max_threads_per_threadgroup();
        let threadgroup_max = (mtpt.width as usize).max(1);
        println!(
            "DEVICE_CAPS simd_width={simd_width} threadgroup_max={threadgroup_max} \
             probe_pipeline_max_total_threads_per_threadgroup={}",
            pipeline.max_total_threads_per_threadgroup()
        );
        Ok(DeviceCaps {
            simd_width,
            threadgroup_max,
        })
    }

    pub fn run(case: &Case, msl: &str, caps: &DeviceCaps) -> Result<GpuReadback, GpuError> {
        let device = select_device()?;
        let device_name = device.name().to_string();
        let pipeline = compile(&device, msl, &case.kernel_name)?;

        let thread_execution_width = pipeline.thread_execution_width() as u64;
        let pipeline_threadgroup_max = pipeline.max_total_threads_per_threadgroup() as u64;
        let n = case.output_len() as u64;
        // Dispatch at the discovered device threadgroup size. Boundaries in the
        // case list were instantiated from the same number.
        let threadgroup = caps.threadgroup_max as u64;
        if pipeline_threadgroup_max < threadgroup {
            return Err(GpuError::Fail(format!(
                "pipeline max_total_threads_per_threadgroup={pipeline_threadgroup_max} is below \
                 instantiated threadgroup={threadgroup}; device and kernel limits disagree"
            )));
        }
        let groups = (n + threadgroup - 1) / threadgroup;
        println!(
            "DISPATCH n={n} threadgroup={threadgroup} groups={groups} \
             pipeline_max_total_threads_per_threadgroup={pipeline_threadgroup_max} \
             pipeline_thread_execution_width={thread_execution_width}"
        );

        let input_bufs: Vec<Buffer> = case.inputs.iter().map(|x| fill_shared(&device, x)).collect();
        let out_init: Vec<f32> =
            vec![crate::cases::OUTPUT_SENTINEL; case.output_len() + case.preserve_pad];
        let out_buf = fill_shared(&device, &out_init);
        let n_buf = fill_shared_u32(&device, n as u32);

        let queue = device.new_command_queue();
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        for (i, buf) in input_bufs.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buf), 0);
        }
        // Output buffer follows the inputs; num_elements follows the output.
        encoder.set_buffer(input_bufs.len() as u64, Some(&out_buf), 0);
        encoder.set_buffer(input_bufs.len() as u64 + 1, Some(&n_buf), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(threadgroup, 1, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let inputs = case
            .inputs
            .iter()
            .zip(input_bufs.iter())
            .map(|(src, buf)| read_f32(buf, src.len()))
            .collect();

        Ok(GpuReadback {
            device_name,
            device_low_power: device.is_low_power(),
            device_headless: device.is_headless(),
            threadgroup,
            groups,
            pipeline_threadgroup_max,
            thread_execution_width,
            inputs,
            out: read_f32(&out_buf, case.output_len() + case.preserve_pad),
        })
    }
}
