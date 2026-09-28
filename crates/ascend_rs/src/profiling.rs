//! Profiling API for collecting performance data on Ascend NPU.
//!
//! Provides RAII wrappers around the CANN profiling APIs for collecting
//! AI Core metrics, task timing, HCCL traces, and user annotations.
//!
//! # Example
//!
//! ```no_run
//! use ascend_rs::prelude::*;
//! use ascend_rs::profiling::*;
//!
//! // Initialize profiling to an output directory
//! let session = ProfSession::new("/tmp/prof_output").unwrap();
//!
//! // Create config for device 0 with task timing
//! let config = ProfConfig::new(&[0], AicoreMetrics::None, ProfDataType::TASK_TIME).unwrap();
//!
//! // Start profiling
//! config.start().unwrap();
//!
//! // ... run inference workload ...
//!
//! // Stop profiling
//! config.stop().unwrap();
//!
//! // Session finalizes automatically on drop
//! ```

use std::ffi::{CString, c_void};
use std::ptr;

use crate::errors::{AclError, AclInnerError, AclResult, ToAclResult};
use crate::stream::AclStream;

// ---- AI Core Metrics ----

/// AI Core performance metrics to collect during profiling.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AicoreMetrics {
    /// Arithmetic unit utilization.
    ArithmeticUtilization = ascend_sys::prof::ACL_AICORE_ARITHMETIC_UTILIZATION,
    /// Pipeline utilization.
    PipeUtilization = ascend_sys::prof::ACL_AICORE_PIPE_UTILIZATION,
    /// Memory bandwidth utilization.
    MemoryBandwidth = ascend_sys::prof::ACL_AICORE_MEMORY_BANDWIDTH,
    /// L0B bandwidth.
    L0bBandwidth = ascend_sys::prof::ACL_AICORE_L0B_AND_WIDTH,
    /// Resource conflict ratio.
    ResourceConflictRatio = ascend_sys::prof::ACL_AICORE_RESOURCE_CONFLICT_RATIO,
    /// Unified Buffer memory usage.
    MemoryUb = ascend_sys::prof::ACL_AICORE_MEMORY_UB,
    /// L2 cache metrics.
    L2Cache = ascend_sys::prof::ACL_AICORE_L2_CACHE,
    /// Pipeline execution utilization.
    PipeExecuteUtilization = ascend_sys::prof::ACL_AICORE_PIPE_EXECUTE_UTILIZATION,
    /// Memory access metrics.
    MemoryAccess = ascend_sys::prof::ACL_AICORE_MEMORY_ACCESS,
    /// No AI Core metrics.
    None = ascend_sys::prof::ACL_AICORE_NONE,
}

// ---- Profiling data type flags ----

/// Bitmask flags selecting which profiling data to collect.
///
/// Combine with bitwise OR: `ProfDataType::ACL_API | ProfDataType::TASK_TIME`
pub struct ProfDataType;

impl ProfDataType {
    /// ACL API call tracing.
    pub const ACL_API: u64 = ascend_sys::prof::ACL_PROF_ACL_API;
    /// Task execution time.
    pub const TASK_TIME: u64 = ascend_sys::prof::ACL_PROF_TASK_TIME;
    /// AI Core metrics (requires [`AicoreMetrics`] != None).
    pub const AICORE_METRICS: u64 = ascend_sys::prof::ACL_PROF_AICORE_METRICS;
    /// AI CPU operator tracing.
    pub const AICPU: u64 = ascend_sys::prof::ACL_PROF_AICPU;
    /// L2 cache metrics.
    pub const L2CACHE: u64 = ascend_sys::prof::ACL_PROF_L2CACHE;
    /// HCCL communication tracing.
    pub const HCCL_TRACE: u64 = ascend_sys::prof::ACL_PROF_HCCL_TRACE;
    /// Training trace data.
    pub const TRAINING_TRACE: u64 = ascend_sys::prof::ACL_PROF_TRAINING_TRACE;
    /// MsProfTx user annotation tracing.
    pub const MSPROFTX: u64 = ascend_sys::prof::ACL_PROF_MSPROFTX;
    /// Runtime API tracing.
    pub const RUNTIME_API: u64 = ascend_sys::prof::ACL_PROF_RUNTIME_API;
    /// L0 task time.
    pub const TASK_TIME_L0: u64 = ascend_sys::prof::ACL_PROF_TASK_TIME_L0;
    /// Task memory usage.
    pub const TASK_MEMORY: u64 = ascend_sys::prof::ACL_PROF_TASK_MEMORY;
    /// Operator attributes.
    pub const OP_ATTR: u64 = ascend_sys::prof::ACL_PROF_OP_ATTR;
}

// ---- Profiling Session (init/finalize lifecycle) ----

/// RAII wrapper for the profiling session lifecycle.
///
/// Calls `aclprofInit` on creation and `aclprofFinalize` on drop.
/// Only one session can be active at a time.
pub struct ProfSession {
    _private: (),
}

impl ProfSession {
    /// Initialize the profiling subsystem.
    ///
    /// `output_path` is the directory where profiling results will be saved.
    pub fn new(output_path: &str) -> AclResult<Self> {
        let c_path = CString::new(output_path).expect("invalid profiling path");
        unsafe {
            ascend_sys::prof::aclprofInit(c_path.as_ptr(), output_path.len()).to_result()?;
        }
        Ok(Self { _private: () })
    }
}

impl Drop for ProfSession {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::prof::aclprofFinalize();
        }
    }
}

// ---- Profiling Config ----

/// Profiling configuration specifying which devices and data to collect.
pub struct ProfConfig {
    config: *mut c_void,
}

impl ProfConfig {
    /// Create a profiling configuration.
    ///
    /// # Arguments
    ///
    /// * `device_ids` - Slice of device IDs to profile
    /// * `aicore_metrics` - Which AI Core metric to collect
    /// * `data_type_config` - Bitmask of [`ProfDataType`] flags
    pub fn new(
        device_ids: &[u32],
        aicore_metrics: AicoreMetrics,
        data_type_config: u64,
    ) -> AclResult<Self> {
        let mut ids = device_ids.to_vec();
        let config = unsafe {
            ascend_sys::prof::aclprofCreateConfig(
                ids.as_mut_ptr(),
                ids.len() as u32,
                aicore_metrics as u32,
                ptr::null(),
                data_type_config,
            )
        };
        if config.is_null() {
            return Err(AclError::Inner(AclInnerError::AclInnerErrorInvalidParam));
        }
        Ok(Self { config })
    }

    /// Start profiling with this configuration.
    pub fn start(&self) -> AclResult<()> {
        unsafe { ascend_sys::prof::aclprofStart(self.config).to_result() }
    }

    /// Stop profiling with this configuration.
    pub fn stop(&self) -> AclResult<()> {
        unsafe { ascend_sys::prof::aclprofStop(self.config).to_result() }
    }
}

impl Drop for ProfConfig {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::prof::aclprofDestroyConfig(self.config);
        }
    }
}

// ---- Step Info (training step tracking) ----

/// Tracks training step boundaries for the profiler.
///
/// Use this to mark step start/end timestamps on a stream so the profiler
/// can correlate performance data with training iterations.
pub struct StepInfo {
    info: *mut c_void,
}

impl StepInfo {
    /// Create a new step info tracker.
    pub fn new() -> AclResult<Self> {
        let info = unsafe { ascend_sys::prof::aclprofCreateStepInfo() };
        if info.is_null() {
            return Err(AclError::Inner(
                AclInnerError::AclInnerErrorRtMemoryAllocation,
            ));
        }
        Ok(Self { info })
    }

    /// Record the start of a training step on the given stream.
    pub fn step_start(&mut self, stream: &AclStream) -> AclResult<()> {
        unsafe {
            ascend_sys::prof::aclprofGetStepTimestamp(
                self.info,
                ascend_sys::prof::ACL_STEP_START,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Record the end of a training step on the given stream.
    pub fn step_end(&mut self, stream: &AclStream) -> AclResult<()> {
        unsafe {
            ascend_sys::prof::aclprofGetStepTimestamp(
                self.info,
                ascend_sys::prof::ACL_STEP_END,
                stream.to_raw(),
            )
            .to_result()
        }
    }
}

impl Drop for StepInfo {
    fn drop(&mut self) {
        unsafe {
            ascend_sys::prof::aclprofDestroyStepInfo(self.info);
        }
    }
}

// ---- User Annotations (MsProfTx) ----

/// A profiling stamp for marking regions or points of interest.
///
/// Used with push/pop (nested regions) or range start/stop (non-nested regions).
pub struct ProfStamp {
    stamp: *mut c_void,
}

impl ProfStamp {
    /// Create a new profiling stamp.
    pub fn new() -> AclResult<Self> {
        let stamp = unsafe { ascend_sys::prof::aclprofCreateStamp() };
        if stamp.is_null() {
            return Err(AclError::Inner(
                AclInnerError::AclInnerErrorRtMemoryAllocation,
            ));
        }
        Ok(Self { stamp })
    }

    /// Set a trace message on this stamp.
    pub fn set_message(&mut self, msg: &str) -> AclResult<()> {
        let c_msg = CString::new(msg).expect("invalid profiling message");
        unsafe {
            ascend_sys::prof::aclprofSetStampTraceMessage(
                self.stamp,
                c_msg.as_ptr(),
                msg.len() as u32,
            )
            .to_result()
        }
    }

    /// Push this stamp onto the profiling stack (start a nested region).
    pub fn push(&self) -> AclResult<()> {
        unsafe { ascend_sys::prof::aclprofPush(self.stamp).to_result() }
    }

    /// Pop the most recent stamp from the profiling stack (end a nested region).
    pub fn pop() -> AclResult<()> {
        unsafe { ascend_sys::prof::aclprofPop().to_result() }
    }

    /// Start a non-nested profiling range.
    ///
    /// Returns a range ID that must be passed to [`ProfStamp::range_stop`].
    pub fn range_start(&self) -> AclResult<u32> {
        let mut range_id: u32 = 0;
        unsafe {
            ascend_sys::prof::aclprofRangeStart(self.stamp, &mut range_id).to_result()?;
        }
        Ok(range_id)
    }

    /// Stop a non-nested profiling range.
    pub fn range_stop(range_id: u32) -> AclResult<()> {
        unsafe { ascend_sys::prof::aclprofRangeStop(range_id).to_result() }
    }

    /// Record a point-in-time mark.
    pub fn mark(&self) -> AclResult<()> {
        unsafe { ascend_sys::prof::aclprofMark(self.stamp).to_result() }
    }
}

impl Drop for ProfStamp {
    fn drop(&mut self) {
        unsafe {
            ascend_sys::prof::aclprofDestroyStamp(self.stamp);
        }
    }
}

/// Record a point-in-time mark with a message on a specific stream.
pub fn prof_mark_ex(msg: &str, stream: &AclStream) -> AclResult<()> {
    let c_msg = CString::new(msg).expect("invalid profiling message");
    unsafe {
        ascend_sys::prof::aclprofMarkEx(c_msg.as_ptr(), msg.len(), stream.to_raw()).to_result()
    }
}
