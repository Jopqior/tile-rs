//! HCCL (Huawei Collective Communication Library) for multi-device communication.
//!
//! Provides safe wrappers around HCCL collective operations including AllReduce,
//! Broadcast, AllGather, ReduceScatter, and point-to-point Send/Recv.
//!
//! # Example
//!
//! ```no_run
//! use ascend_rs::prelude::*;
//! use ascend_rs::hccl::*;
//!
//! let acl = Acl::new().unwrap();
//! let device = Device::new(&acl).unwrap();
//! let context = AclContext::new(&device).unwrap();
//! let stream = AclStream::new(&context).unwrap();
//!
//! // Initialize communicator (rank 0 of 2 devices)
//! let root_info = HcclRootInfo::new().unwrap();
//! let comm = Communicator::from_root_info(2, &root_info, 0).unwrap();
//!
//! // Perform collective operations...
//! // comm.all_reduce(send, recv, count, DataType::Fp32, ReduceOp::Sum, &stream);
//! ```

use std::ffi::c_void;
use std::fmt;

use crate::stream::AclStream;

// ---- Error handling ----

/// HCCL error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcclError(pub u32);

impl HcclError {
    pub fn is_success(&self) -> bool {
        self.0 == ascend_sys::hccl::HCCL_SUCCESS
    }
}

impl fmt::Display for HcclError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self.0 {
            ascend_sys::hccl::HCCL_SUCCESS => "success",
            ascend_sys::hccl::HCCL_E_PARA => "invalid parameter",
            ascend_sys::hccl::HCCL_E_PTR => "invalid pointer",
            ascend_sys::hccl::HCCL_E_MEMORY => "memory error",
            ascend_sys::hccl::HCCL_E_INTERNAL => "internal error",
            ascend_sys::hccl::HCCL_E_NOT_SUPPORT => "not supported",
            ascend_sys::hccl::HCCL_E_NOT_FOUND => "not found",
            ascend_sys::hccl::HCCL_E_UNAVAIL => "unavailable",
            ascend_sys::hccl::HCCL_E_SYSCALL => "syscall error",
            ascend_sys::hccl::HCCL_E_TIMEOUT => "timeout",
            ascend_sys::hccl::HCCL_E_RUNTIME => "runtime error",
            ascend_sys::hccl::HCCL_E_NETWORK => "network error",
            _ => "unknown HCCL error",
        };
        write!(f, "HCCL error {}: {}", self.0, msg)
    }
}

impl std::error::Error for HcclError {}

pub type HcclResult<T> = Result<T, HcclError>;

fn check_hccl(result: u32) -> HcclResult<()> {
    if result == ascend_sys::hccl::HCCL_SUCCESS {
        Ok(())
    } else {
        Err(HcclError(result))
    }
}

// ---- Reduction operation ----

/// Reduction operation for collective communication.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum = ascend_sys::hccl::HCCL_REDUCE_SUM,
    /// Element-wise product.
    Prod = ascend_sys::hccl::HCCL_REDUCE_PROD,
    /// Element-wise maximum.
    Max = ascend_sys::hccl::HCCL_REDUCE_MAX,
    /// Element-wise minimum.
    Min = ascend_sys::hccl::HCCL_REDUCE_MIN,
}

// ---- Data type ----

/// Data types for HCCL communication buffers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int8 = ascend_sys::hccl::HCCL_DATA_TYPE_INT8,
    Int16 = ascend_sys::hccl::HCCL_DATA_TYPE_INT16,
    Int32 = ascend_sys::hccl::HCCL_DATA_TYPE_INT32,
    Fp16 = ascend_sys::hccl::HCCL_DATA_TYPE_FP16,
    Fp32 = ascend_sys::hccl::HCCL_DATA_TYPE_FP32,
    Int64 = ascend_sys::hccl::HCCL_DATA_TYPE_INT64,
    Uint64 = ascend_sys::hccl::HCCL_DATA_TYPE_UINT64,
    Uint8 = ascend_sys::hccl::HCCL_DATA_TYPE_UINT8,
    Uint16 = ascend_sys::hccl::HCCL_DATA_TYPE_UINT16,
    Uint32 = ascend_sys::hccl::HCCL_DATA_TYPE_UINT32,
    Fp64 = ascend_sys::hccl::HCCL_DATA_TYPE_FP64,
    Bfp16 = ascend_sys::hccl::HCCL_DATA_TYPE_BFP16,
    Int128 = ascend_sys::hccl::HCCL_DATA_TYPE_INT128,
}

// ---- Root Info ----

/// Opaque bootstrap information for initializing communicators.
///
/// One rank generates the root info and distributes it to all other ranks
/// (e.g., via MPI or sockets) before calling [`Communicator::from_root_info`].
pub struct HcclRootInfo {
    inner: ascend_sys::hccl::HcclRootInfo,
}

impl HcclRootInfo {
    /// Generate new root info for bootstrapping a communicator.
    ///
    /// This should be called on one rank, then the root info should be
    /// distributed to all participating ranks.
    pub fn new() -> HcclResult<Self> {
        let mut info = ascend_sys::hccl::HcclRootInfo {
            internal: [0u8; ascend_sys::hccl::HCCL_ROOT_INFO_BYTES],
        };
        check_hccl(unsafe { ascend_sys::hccl::HcclGetRootInfo(&mut info) })?;
        Ok(Self { inner: info })
    }

    /// Create root info from raw bytes (received from another rank).
    pub fn from_bytes(bytes: &[u8; ascend_sys::hccl::HCCL_ROOT_INFO_BYTES]) -> Self {
        Self {
            inner: ascend_sys::hccl::HcclRootInfo { internal: *bytes },
        }
    }

    /// Get the raw bytes for transmission to other ranks.
    pub fn as_bytes(&self) -> &[u8; ascend_sys::hccl::HCCL_ROOT_INFO_BYTES] {
        &self.inner.internal
    }
}

// ---- Communicator ----

/// RAII wrapper for an HCCL communicator.
///
/// A communicator represents a group of devices that can participate in
/// collective operations. It is automatically destroyed on drop.
pub struct Communicator {
    comm: ascend_sys::hccl::HcclComm,
}

impl Communicator {
    /// Initialize a communicator using root info.
    ///
    /// # Arguments
    ///
    /// * `n_ranks` - Total number of ranks in the communicator
    /// * `root_info` - Bootstrap info generated by [`HcclRootInfo::new`]
    /// * `rank` - This process's rank (0-based)
    pub fn from_root_info(n_ranks: u32, root_info: &HcclRootInfo, rank: u32) -> HcclResult<Self> {
        let mut comm: ascend_sys::hccl::HcclComm = std::ptr::null_mut();
        check_hccl(unsafe {
            ascend_sys::hccl::HcclCommInitRootInfo(n_ranks, &root_info.inner, rank, &mut comm)
        })?;
        Ok(Self { comm })
    }

    /// Get the number of ranks in this communicator.
    pub fn rank_size(&self) -> HcclResult<u32> {
        let mut size: u32 = 0;
        check_hccl(unsafe { ascend_sys::hccl::HcclGetRankSize(self.comm, &mut size) })?;
        Ok(size)
    }

    /// Get this process's rank in the communicator.
    pub fn rank_id(&self) -> HcclResult<u32> {
        let mut rank: u32 = 0;
        check_hccl(unsafe { ascend_sys::hccl::HcclGetRankId(self.comm, &mut rank) })?;
        Ok(rank)
    }

    /// Check for asynchronous errors on this communicator.
    pub fn async_error(&self) -> HcclResult<()> {
        let mut err: u32 = 0;
        check_hccl(unsafe { ascend_sys::hccl::HcclGetCommAsyncError(self.comm, &mut err) })?;
        check_hccl(err)
    }

    // ---- Collective Operations ----
    // All collective operations are asynchronous and execute on the given stream.

    /// AllReduce: reduce data across all ranks and distribute the result.
    ///
    /// # Safety
    ///
    /// `send_buf` and `recv_buf` must point to valid device memory of at least
    /// `count` elements of the given data type.
    pub unsafe fn all_reduce(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        count: u64,
        data_type: DataType,
        op: ReduceOp,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclAllReduce(
                send_buf,
                recv_buf,
                count,
                data_type as u32,
                op as u32,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    /// Broadcast: send data from `root` rank to all other ranks.
    ///
    /// # Safety
    ///
    /// `buf` must point to valid device memory of at least `count` elements.
    pub unsafe fn broadcast(
        &self,
        buf: *mut c_void,
        count: u64,
        data_type: DataType,
        root: u32,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclBroadcast(
                buf,
                count,
                data_type as u32,
                root,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    /// ReduceScatter: reduce data then scatter equal chunks to each rank.
    ///
    /// # Safety
    ///
    /// `send_buf` must have `recv_count * n_ranks` elements.
    /// `recv_buf` must have `recv_count` elements.
    pub unsafe fn reduce_scatter(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        recv_count: u64,
        data_type: DataType,
        op: ReduceOp,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclReduceScatter(
                send_buf,
                recv_buf,
                recv_count,
                data_type as u32,
                op as u32,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    /// AllGather: gather data from all ranks and distribute the concatenated result.
    ///
    /// # Safety
    ///
    /// `send_buf` must have `send_count` elements.
    /// `recv_buf` must have `send_count * n_ranks` elements.
    pub unsafe fn all_gather(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        send_count: u64,
        data_type: DataType,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclAllGather(
                send_buf,
                recv_buf,
                send_count,
                data_type as u32,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    /// Reduce: reduce data from all ranks to a single `root` rank.
    ///
    /// # Safety
    ///
    /// Both buffers must have `count` elements. Only `root` rank needs
    /// a valid `recv_buf`.
    pub unsafe fn reduce(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        count: u64,
        data_type: DataType,
        op: ReduceOp,
        root: u32,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclReduce(
                send_buf,
                recv_buf,
                count,
                data_type as u32,
                op as u32,
                root,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    /// Scatter: distribute chunks of data from `root` to all ranks.
    ///
    /// # Safety
    ///
    /// `send_buf` on root must have `recv_count * n_ranks` elements.
    /// `recv_buf` must have `recv_count` elements.
    pub unsafe fn scatter(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        recv_count: u64,
        data_type: DataType,
        root: u32,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclScatter(
                send_buf,
                recv_buf,
                recv_count,
                data_type as u32,
                root,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    /// Gather: collect data from all ranks to `root`.
    ///
    /// # Safety
    ///
    /// `send_buf` must have `send_count` elements.
    /// `recv_buf` on root must have `send_count * n_ranks` elements.
    pub unsafe fn gather(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        send_count: u64,
        data_type: DataType,
        root: u32,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclGather(
                send_buf,
                recv_buf,
                send_count,
                data_type as u32,
                root,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    /// AlltoAll: each rank sends equal-sized data to every other rank.
    ///
    /// # Safety
    ///
    /// `send_buf` must have `send_count * n_ranks` elements.
    /// `recv_buf` must have `recv_count * n_ranks` elements.
    pub unsafe fn all_to_all(
        &self,
        send_buf: *const c_void,
        send_count: u64,
        send_type: DataType,
        recv_buf: *mut c_void,
        recv_count: u64,
        recv_type: DataType,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclAlltoAll(
                send_buf,
                send_count,
                send_type as u32,
                recv_buf,
                recv_count,
                recv_type as u32,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    // ---- Point-to-point Operations ----

    /// Send data to a specific rank.
    ///
    /// # Safety
    ///
    /// `send_buf` must point to valid device memory of at least `count` elements.
    pub unsafe fn send(
        &self,
        send_buf: *const c_void,
        count: u64,
        data_type: DataType,
        dest_rank: u32,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclSend(
                send_buf,
                count,
                data_type as u32,
                dest_rank,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    /// Receive data from a specific rank.
    ///
    /// # Safety
    ///
    /// `recv_buf` must point to valid device memory of at least `count` elements.
    pub unsafe fn recv(
        &self,
        recv_buf: *mut c_void,
        count: u64,
        data_type: DataType,
        src_rank: u32,
        stream: &AclStream,
    ) -> HcclResult<()> {
        check_hccl(unsafe {
            ascend_sys::hccl::HcclRecv(
                recv_buf,
                count,
                data_type as u32,
                src_rank,
                self.comm,
                stream.to_raw(),
            )
        })
    }

    // ---- Synchronization ----

    /// Barrier synchronization across all ranks.
    pub fn barrier(&self, stream: &AclStream) -> HcclResult<()> {
        check_hccl(unsafe { ascend_sys::hccl::HcclBarrier(self.comm, stream.to_raw()) })
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::hccl::HcclCommDestroy(self.comm);
        }
    }
}

/// Initialize communicators for all local devices.
///
/// Returns one communicator per device. Equivalent to `HcclCommInitAll`.
///
/// # Arguments
///
/// * `device_ids` - Slice of device IDs to initialize
pub fn init_all(device_ids: &[i32]) -> HcclResult<Vec<Communicator>> {
    let n = device_ids.len() as u32;
    let mut ids = device_ids.to_vec();
    let mut raw_comms: Vec<ascend_sys::hccl::HcclComm> =
        vec![std::ptr::null_mut(); device_ids.len()];
    check_hccl(unsafe {
        ascend_sys::hccl::HcclCommInitAll(n, ids.as_mut_ptr(), raw_comms.as_mut_ptr())
    })?;
    Ok(raw_comms
        .into_iter()
        .map(|comm| Communicator { comm })
        .collect())
}
