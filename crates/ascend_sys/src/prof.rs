//! Manual FFI declarations for CANN profiling APIs.
//!
//! These declarations match `acl/acl_prof.h` from the CANN SDK.
//! They can be replaced with bindgen-generated bindings once the header
//! is added to the wrapper.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::c_void;
use std::os::raw::c_char;

// ---- Opaque types ----

pub type aclprofConfig = c_void;
pub type aclprofStopConfig = c_void;
pub type aclprofAicoreEvents = c_void;
pub type aclprofSubscribeConfig = c_void;
pub type aclprofStepInfo = c_void;

pub type aclrtStream = *mut c_void;

// ---- Data type flags (bitmask) ----

pub const ACL_PROF_ACL_API: u64 = 0x0000_0001;
pub const ACL_PROF_TASK_TIME: u64 = 0x0000_0002;
pub const ACL_PROF_AICORE_METRICS: u64 = 0x0000_0004;
pub const ACL_PROF_AICPU: u64 = 0x0000_0008;
pub const ACL_PROF_L2CACHE: u64 = 0x0000_0010;
pub const ACL_PROF_HCCL_TRACE: u64 = 0x0000_0020;
pub const ACL_PROF_TRAINING_TRACE: u64 = 0x0000_0040;
pub const ACL_PROF_MSPROFTX: u64 = 0x0000_0080;
pub const ACL_PROF_RUNTIME_API: u64 = 0x0000_0100;
pub const ACL_PROF_TASK_TIME_L0: u64 = 0x0000_0800;
pub const ACL_PROF_TASK_MEMORY: u64 = 0x0000_1000;
pub const ACL_PROF_OP_ATTR: u64 = 0x0000_4000;

// ---- Enums ----

pub type aclprofAicoreMetrics = u32;
pub const ACL_AICORE_ARITHMETIC_UTILIZATION: aclprofAicoreMetrics = 0;
pub const ACL_AICORE_PIPE_UTILIZATION: aclprofAicoreMetrics = 1;
pub const ACL_AICORE_MEMORY_BANDWIDTH: aclprofAicoreMetrics = 2;
pub const ACL_AICORE_L0B_AND_WIDTH: aclprofAicoreMetrics = 3;
pub const ACL_AICORE_RESOURCE_CONFLICT_RATIO: aclprofAicoreMetrics = 4;
pub const ACL_AICORE_MEMORY_UB: aclprofAicoreMetrics = 5;
pub const ACL_AICORE_L2_CACHE: aclprofAicoreMetrics = 6;
pub const ACL_AICORE_PIPE_EXECUTE_UTILIZATION: aclprofAicoreMetrics = 7;
pub const ACL_AICORE_MEMORY_ACCESS: aclprofAicoreMetrics = 8;
pub const ACL_AICORE_NONE: aclprofAicoreMetrics = 0xFF;

pub type aclprofStepTag = u32;
pub const ACL_STEP_START: aclprofStepTag = 0;
pub const ACL_STEP_END: aclprofStepTag = 1;

extern "C" {
    // ---- Lifecycle ----
    pub fn aclprofInit(profilerResultPath: *const c_char, length: usize) -> i32;
    pub fn aclprofFinalize() -> i32;
    pub fn aclprofStart(profilerConfig: *const aclprofConfig) -> i32;
    pub fn aclprofStop(profilerConfig: *const aclprofConfig) -> i32;

    // ---- Config ----
    pub fn aclprofCreateConfig(
        deviceIdList: *mut u32,
        deviceNums: u32,
        aicoreMetrics: aclprofAicoreMetrics,
        aicoreEvents: *const aclprofAicoreEvents,
        dataTypeConfig: u64,
    ) -> *mut aclprofConfig;
    pub fn aclprofDestroyConfig(profilerConfig: *const aclprofConfig) -> i32;

    // ---- Step tracking ----
    pub fn aclprofCreateStepInfo() -> *mut aclprofStepInfo;
    pub fn aclprofDestroyStepInfo(stepinfo: *mut aclprofStepInfo);
    pub fn aclprofGetStepTimestamp(
        stepInfo: *mut aclprofStepInfo,
        tag: aclprofStepTag,
        stream: aclrtStream,
    ) -> i32;

    // ---- Model subscription ----
    pub fn aclprofModelSubscribe(
        modelId: u32,
        profSubscribeConfig: *const aclprofSubscribeConfig,
    ) -> i32;
    pub fn aclprofModelUnSubscribe(modelId: u32) -> i32;
    pub fn aclprofCreateSubscribeConfig(
        timeInfoSwitch: i8,
        aicoreMetrics: aclprofAicoreMetrics,
        fd: *mut c_void,
    ) -> *mut aclprofSubscribeConfig;
    pub fn aclprofDestroySubscribeConfig(profSubscribeConfig: *const aclprofSubscribeConfig)
        -> i32;

    // ---- Subscription data queries ----
    pub fn aclprofGetOpNum(opInfo: *const c_void, opInfoLen: usize, opNumber: *mut u32) -> i32;
    pub fn aclprofGetOpType(
        opInfo: *const c_void,
        opInfoLen: usize,
        index: u32,
        opType: *mut c_char,
        opTypeLen: usize,
    ) -> i32;
    pub fn aclprofGetOpName(
        opInfo: *const c_void,
        opInfoLen: usize,
        index: u32,
        opName: *mut c_char,
        opNameLen: usize,
    ) -> i32;
    pub fn aclprofGetOpStart(opInfo: *const c_void, opInfoLen: usize, index: u32) -> u64;
    pub fn aclprofGetOpEnd(opInfo: *const c_void, opInfoLen: usize, index: u32) -> u64;
    pub fn aclprofGetOpDuration(opInfo: *const c_void, opInfoLen: usize, index: u32) -> u64;

    // ---- User annotations (MsProfTx) ----
    pub fn aclprofCreateStamp() -> *mut c_void;
    pub fn aclprofDestroyStamp(stamp: *mut c_void);
    pub fn aclprofSetStampTraceMessage(stamp: *mut c_void, msg: *const c_char, msgLen: u32) -> i32;
    pub fn aclprofPush(stamp: *mut c_void) -> i32;
    pub fn aclprofPop() -> i32;
    pub fn aclprofRangeStart(stamp: *mut c_void, rangeId: *mut u32) -> i32;
    pub fn aclprofRangeStop(rangeId: u32) -> i32;
    pub fn aclprofMark(stamp: *mut c_void) -> i32;
    pub fn aclprofMarkEx(msg: *const c_char, msgLen: usize, stream: aclrtStream) -> i32;
}
