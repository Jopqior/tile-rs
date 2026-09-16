//! Manual FFI declarations for HCCL (Huawei Collective Communication Library).
//!
//! These declarations match `hccl/hccl.h` and `hccl/hccl_types.h` from the CANN SDK.
//! They can be replaced with bindgen-generated bindings once the headers
//! are added to the wrapper.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::c_void;

// ---- Result type ----

pub type HcclResult = u32;
pub const HCCL_SUCCESS: HcclResult = 0;
pub const HCCL_E_PARA: HcclResult = 1;
pub const HCCL_E_PTR: HcclResult = 2;
pub const HCCL_E_MEMORY: HcclResult = 3;
pub const HCCL_E_INTERNAL: HcclResult = 4;
pub const HCCL_E_NOT_SUPPORT: HcclResult = 5;
pub const HCCL_E_NOT_FOUND: HcclResult = 6;
pub const HCCL_E_UNAVAIL: HcclResult = 7;
pub const HCCL_E_SYSCALL: HcclResult = 8;
pub const HCCL_E_TIMEOUT: HcclResult = 9;
pub const HCCL_E_OPEN_FILE_FAILURE: HcclResult = 10;
pub const HCCL_E_TCP_CONNECT: HcclResult = 11;
pub const HCCL_E_ROCE_CONNECT: HcclResult = 12;
pub const HCCL_E_TCP_TRANSFER: HcclResult = 13;
pub const HCCL_E_ROCE_TRANSFER: HcclResult = 14;
pub const HCCL_E_RUNTIME: HcclResult = 15;
pub const HCCL_E_DRV: HcclResult = 16;
pub const HCCL_E_PROFILING: HcclResult = 17;
pub const HCCL_E_CCE: HcclResult = 18;
pub const HCCL_E_NETWORK: HcclResult = 19;
pub const HCCL_E_AGAIN: HcclResult = 20;
pub const HCCL_E_REMOTE: HcclResult = 21;
pub const HCCL_E_SUSPENDING: HcclResult = 22;

// ---- Communicator handle ----

pub type HcclComm = *mut c_void;

// ---- Reduction operations ----

pub type HcclReduceOp = u32;
pub const HCCL_REDUCE_SUM: HcclReduceOp = 0;
pub const HCCL_REDUCE_PROD: HcclReduceOp = 1;
pub const HCCL_REDUCE_MAX: HcclReduceOp = 2;
pub const HCCL_REDUCE_MIN: HcclReduceOp = 3;

// ---- Data types ----

pub type HcclDataType = u32;
pub const HCCL_DATA_TYPE_INT8: HcclDataType = 0;
pub const HCCL_DATA_TYPE_INT16: HcclDataType = 1;
pub const HCCL_DATA_TYPE_INT32: HcclDataType = 2;
pub const HCCL_DATA_TYPE_FP16: HcclDataType = 3;
pub const HCCL_DATA_TYPE_FP32: HcclDataType = 4;
pub const HCCL_DATA_TYPE_INT64: HcclDataType = 5;
pub const HCCL_DATA_TYPE_UINT64: HcclDataType = 6;
pub const HCCL_DATA_TYPE_UINT8: HcclDataType = 7;
pub const HCCL_DATA_TYPE_UINT16: HcclDataType = 8;
pub const HCCL_DATA_TYPE_UINT32: HcclDataType = 9;
pub const HCCL_DATA_TYPE_FP64: HcclDataType = 10;
pub const HCCL_DATA_TYPE_BFP16: HcclDataType = 11;
pub const HCCL_DATA_TYPE_INT128: HcclDataType = 12;

// ---- Root info ----

pub const HCCL_ROOT_INFO_BYTES: usize = 4108;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct HcclRootInfo {
    pub internal: [u8; HCCL_ROOT_INFO_BYTES],
}

// ---- Send/Recv types (for batch operations) ----

pub type HcclSendRecvType = u32;
pub const HCCL_SEND: HcclSendRecvType = 0;
pub const HCCL_RECV: HcclSendRecvType = 1;

#[repr(C)]
pub struct HcclSendRecvItem {
    pub sendRecvType: HcclSendRecvType,
    pub buf: *mut c_void,
    pub count: u64,
    pub dataType: HcclDataType,
    pub remoteRank: u32,
}

// aclrtStream from core bindings
pub type aclrtStream = *mut c_void;

extern "C" {
    // ---- Initialization ----
    pub fn HcclGetRootInfo(rootInfo: *mut HcclRootInfo) -> HcclResult;
    pub fn HcclCommInitRootInfo(
        nRanks: u32,
        rootInfo: *const HcclRootInfo,
        rank: u32,
        comm: *mut HcclComm,
    ) -> HcclResult;
    pub fn HcclCommInitAll(ndev: u32, devices: *mut i32, comms: *mut HcclComm) -> HcclResult;

    // ---- Collective operations ----
    pub fn HcclAllReduce(
        sendBuf: *const c_void,
        recvBuf: *mut c_void,
        count: u64,
        dataType: HcclDataType,
        op: HcclReduceOp,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    pub fn HcclBroadcast(
        buf: *mut c_void,
        count: u64,
        dataType: HcclDataType,
        root: u32,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    pub fn HcclReduceScatter(
        sendBuf: *const c_void,
        recvBuf: *mut c_void,
        recvCount: u64,
        dataType: HcclDataType,
        op: HcclReduceOp,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    pub fn HcclAllGather(
        sendBuf: *const c_void,
        recvBuf: *mut c_void,
        sendCount: u64,
        dataType: HcclDataType,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    pub fn HcclReduce(
        sendBuf: *const c_void,
        recvBuf: *mut c_void,
        count: u64,
        dataType: HcclDataType,
        op: HcclReduceOp,
        root: u32,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    pub fn HcclScatter(
        sendBuf: *const c_void,
        recvBuf: *mut c_void,
        recvCount: u64,
        dataType: HcclDataType,
        root: u32,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    pub fn HcclGather(
        sendBuf: *const c_void,
        recvBuf: *mut c_void,
        sendCount: u64,
        dataType: HcclDataType,
        root: u32,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    pub fn HcclAlltoAll(
        sendBuf: *const c_void,
        sendCount: u64,
        sendType: HcclDataType,
        recvBuf: *mut c_void,
        recvCount: u64,
        recvType: HcclDataType,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    // ---- Point-to-point ----
    pub fn HcclSend(
        sendBuf: *const c_void,
        count: u64,
        dataType: HcclDataType,
        destRank: u32,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    pub fn HcclRecv(
        recvBuf: *mut c_void,
        count: u64,
        dataType: HcclDataType,
        srcRank: u32,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    pub fn HcclBatchSendRecv(
        sendRecvInfo: *mut HcclSendRecvItem,
        itemNum: u32,
        comm: HcclComm,
        stream: aclrtStream,
    ) -> HcclResult;

    // ---- Synchronization ----
    pub fn HcclBarrier(comm: HcclComm, stream: aclrtStream) -> HcclResult;

    // ---- Communicator management ----
    pub fn HcclCommDestroy(comm: HcclComm) -> HcclResult;
    pub fn HcclGetRankSize(comm: HcclComm, rankSize: *mut u32) -> HcclResult;
    pub fn HcclGetRankId(comm: HcclComm, rank: *mut u32) -> HcclResult;
    pub fn HcclGetCommAsyncError(comm: HcclComm, asyncError: *mut HcclResult) -> HcclResult;
}
