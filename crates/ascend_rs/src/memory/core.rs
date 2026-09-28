use std::ffi::c_void;
use std::ptr;

use super::device_ptr::DevicePtr;
use crate::errors::{AclError, AclInnerError, AclResult, ToAclResult};
use crate::stream::AclStream;
use crate::types::DeviceSend;

/// Memory allocation rules. Precise description can be found here:
/// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/appdevgapi/aclcppdevg_03_1349.html
///
/// If `LowBandWidth` or `HighBandWidth` is configured, the system will internally default
/// to `HugeFirst`, prioritizing the allocation of large pages. If other values ​​are configured
/// besides `LowBandWidth` and `HighBandWidth`, the system will choose to request memory
/// from high-bandwidth or low-bandwidth physical memory based on hardware support.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum AclrtMemMallocPolicy {
    /// When requesting large page memory, the memory allocation granularity is 2MB.
    /// If the allocation is less than a multiple of 2MB, it will be aligned upwards to 2MB.
    ///
    /// When the requested memory is less than or equal to 1MB, even using the same memory allocation
    /// rule, it will request memory in the form of ordinary pages. When the requested memory is
    /// greater than 1MB, it will prioritize requesting memory in the form of large pages.
    HugeFirst = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_MALLOC_HUGE_FIRST,

    /// When requesting large page memory, the memory allocation granularity is 2MB.
    /// If the allocation is less than a multiple of 2MB, it will be aligned upwards to 2MB.
    ///
    /// When this option is configured, it means that only large pages are requested;
    /// if there is not enough large page memory, an error will be returned.
    HugeOnly = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_MALLOC_HUGE_ONLY,

    /// Only request regular pages; if there is not enough memory for regular pages,
    /// return an error.
    NormalOnly = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_MALLOC_NORMAL_ONLY,

    /// In memory copying scenarios between two devices, use this option to request
    /// large page memory. The memory allocation granularity is 2MB, and any allocation
    /// less than a multiple of 2MB will be aligned upwards by 2MB.
    ///
    /// When this option is configured, it means that large page memory is requested first,
    /// and if large page memory is insufficient, ordinary page memory will be used.
    HugeFirstP2P = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_MALLOC_HUGE_FIRST_P2P,

    /// In memory copying scenarios between two devices, use this option to request
    /// large page memory. The memory allocation granularity is 2MB, and any allocation
    /// less than a multiple of 2MB will be aligned upwards by 2MB.
    ///
    /// When this option is configured, it means that only large pages are requested;
    /// if there is not enough large page memory, an error will be returned.
    HugeOnlyP2P = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_MALLOC_HUGE_ONLY_P2P,

    /// Use this option in scenarios involving memory copying between two devices to
    /// indicate that only memory for ordinary pages is requested.
    NormalOnlyP2P = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_MALLOC_NORMAL_ONLY_P2P,

    /// When allocating large page memory, the memory allocation granularity is 1GB.
    /// Quantities less than a multiple of 1GB are aligned upwards by 1GB.
    Huge1GOnly = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_MALLOC_HUGE1G_ONLY,

    /// When copying memory between two devices, use this option to allocate huge pages.
    /// The memory allocation granularity is 1GB, and any amount less than a multiple of
    /// 1GB is aligned upwards by 1GB.
    Huge1GOnlyP2P = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_MALLOC_HUGE1G_ONLY_P2P,

    /// Allocate memory from physical memory with low bandwidth.
    ///
    /// Setting this option is invalid; the system will default to selecting based on
    /// the memory type supported by the hardware.
    LowBandWidth = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_TYPE_LOW_BAND_WIDTH,

    /// Allocate memory from physical memory with high bandwidth.
    ///
    /// Setting this option is invalid; the system will default to selecting based on
    /// the memory type supported by the hardware.
    HighBandWidth = ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_TYPE_HIGH_BAND_WIDTH,

    /// The memory allocated is read-only in user space; any modification to this memory in
    /// user space will result in failure.
    AccessUserSpaceReadonly =
        ascend_sys::core::aclrtMemMallocPolicy_ACL_MEM_ACCESS_USER_SPACE_READONLY,
}

impl From<AclrtMemMallocPolicy> for u32 {
    fn from(value: AclrtMemMallocPolicy) -> Self {
        value as u32
    }
}

/* FIXME: check zst for the API below */

/// Allocate linear memory on the device and return a pointer to it with the memory starting address
/// aligned to 64 bytes.
///
/// The memory allocated by this interface will not initialize the content.
///
/// Memory allocated using the [`device_malloc`] interface needs to be released using the
/// [`device_free`] interface.
///
/// This interface does not perform implicit device or stream synchronization.
/// It will return the result immediately if the memory allocation is successful or fails.
///
/// # Safety
///
/// Caller must ensure that memory is initialized before copying to the host memory. Device
/// pointer can't be dereferences on the host side.
pub unsafe fn device_malloc<T: DeviceSend>(
    size: usize,
    policy: AclrtMemMallocPolicy,
) -> AclResult<DevicePtr<T>> {
    unsafe {
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        let size = std::mem::size_of::<T>() * size;
        ascend_sys::core::aclrtMalloc(&mut ptr, size, policy.into()).to_result()?;
        let ptr = ptr as *mut T;
        DevicePtr::<T>::new(ptr).ok_or(AclError::Inner(
            AclInnerError::AclInnerErrorRtMemoryAllocation,
        ))
    }
}

/// Allocate host memory (this memory is page-locked memory), and the system guarantees
/// that the memory's starting address is aligned to 64 bytes.
///
/// The memory allocated by this interface will not initialize the content.
///
/// Memory allocated using the [`malloc_locked`] interface needs to be released using the
/// [`free_locked`] interface.
///
/// This interface does not perform implicit device or stream synchronization.
/// It will return the result immediately if the memory ,allocation is successful or fails.
///
/// # Safety
///
/// Caller must ensure that memory is initialized before using it.
pub unsafe fn malloc_locked<T: DeviceSend>(size: usize) -> AclResult<*mut T> {
    unsafe {
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        let size = std::mem::size_of::<T>() * size;
        ascend_sys::core::aclrtMallocHost(&mut ptr, size).to_result()?;
        Ok(ptr as *mut T)
    }
}

/// Implement memory copying from host to device.
///
/// This interface will immediately perform a memory copy; no implicit device
/// synchronization or stream synchronization will be performed inside the function.
///
/// # Safety
///
/// The host allocation must have been initialized and have `size` size.
/// The device allocation must have `size` size.
pub unsafe fn host_to_device<T: DeviceSend>(
    src: *const T,
    dst: DevicePtr<T>,
    size: usize,
) -> AclResult<()> {
    unsafe {
        let size = std::mem::size_of::<T>() * size;
        ascend_sys::core::aclrtMemcpy(
            dst.as_ptr() as *mut c_void,
            size,
            src as *const c_void,
            size,
            ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_HOST_TO_DEVICE,
        )
        .to_result()
    }
}

/// Implement memory copying between two host allocations.
///
/// This interface will immediately perform a memory copy; no implicit device
/// synchronization or stream synchronization will be performed inside the function.
///
/// # Safety
///
/// The `src` allocation must have been initialized and have `size` size.
/// The `dst` allocation must have `size` size.
pub unsafe fn host_to_host<T: DeviceSend>(
    src: *const T,
    dst: *mut T,
    size: usize,
) -> AclResult<()> {
    unsafe {
        let size = std::mem::size_of::<T>() * size;
        ascend_sys::core::aclrtMemcpy(
            dst as *mut c_void,
            size,
            src as *const c_void,
            size,
            ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_HOST_TO_HOST,
        )
        .to_result()
    }
}

/// Implement memory copying from device memory to host memory.
///
/// This interface will immediately perform a memory copy; no implicit device
/// synchronization or stream synchronization will be performed inside the function.
///
/// # Safety
///
/// The `src` allocation must have been initialized and have `size` size.
/// The `dst` allocation must have `size` size.
pub unsafe fn device_to_host<T: DeviceSend>(
    src: DevicePtr<T>,
    dst: *mut T,
    size: usize,
) -> AclResult<()> {
    unsafe {
        let size = std::mem::size_of::<T>() * size;
        ascend_sys::core::aclrtMemcpy(
            dst as *mut c_void,
            size,
            src.as_ptr() as *mut c_void,
            size,
            ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_DEVICE_TO_HOST,
        )
        .to_result()
    }
}

/// Implement memory copying between two device allocations.
///
/// This interface will immediately perform a memory copy; no implicit device
/// synchronization or stream synchronization will be performed inside the function.
///
/// # Safety
///
/// The `src` allocation must have been initialized and have `size` size.
/// The `dst` allocation must have `size` size.
/// Data interaction between two devices must be enabled via [`Device::enable_peer_access`].
pub unsafe fn device_to_device<T: DeviceSend>(
    src: DevicePtr<T>,
    dst: DevicePtr<T>,
    size: usize,
) -> AclResult<()> {
    unsafe {
        let size = std::mem::size_of::<T>() * size;
        ascend_sys::core::aclrtMemcpy(
            dst.as_ptr() as *mut c_void,
            size,
            src.as_ptr() as *mut c_void,
            size,
            ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_DEVICE_TO_DEVICE,
        )
        .to_result()
    }
}

/// Release the memory on the host.
///
/// [`free_locked`] interface can only release memory allocated through the
/// [`malloc_locked`].
///
/// This interface will immediately release the passed-in memory, and the function
/// will not perform implicit device synchronization or stream synchronization.
///
/// Caller need to ensure that no other pointer to the deallocated memory exists.
pub unsafe fn free_locked<T: DeviceSend>(ptr: *mut T) -> AclResult<()> {
    unsafe { ascend_sys::core::aclrtFreeHost(ptr as *mut c_void).to_result() }
}

/// Release the memory on the device.
///
/// [`device_free`] interface can only release memory allocated through the
/// [`device_malloc`].
///
/// This interface will immediately release the passed-in memory, and the function
/// will not perform implicit device synchronization or stream synchronization.
///
/// # Safety
///
/// Caller need to ensure that no other pointer to the deallocated memory exists.
pub unsafe fn device_free<T: DeviceSend>(ptr: DevicePtr<T>) -> AclResult<()> {
    unsafe { ascend_sys::core::aclrtFree(ptr.as_ptr() as *mut c_void).to_result() }
}

// ---- Async (stream-ordered) memory operations ----

/// Asynchronously copy memory from host to device on the given stream.
///
/// The copy is enqueued on the stream and will execute asynchronously.
/// The host memory (`src`) must remain valid until the copy completes
/// (use [`AclStream::synchronize`] or events to ensure this).
///
/// For best performance, `src` should be page-locked memory allocated via
/// [`malloc_locked`].
///
/// # Safety
///
/// The host allocation must have been initialized and have `size` elements.
/// The device allocation must have space for `size` elements.
/// The host memory must not be freed until the stream operation completes.
pub unsafe fn host_to_device_async<T: DeviceSend>(
    src: *const T,
    dst: DevicePtr<T>,
    size: usize,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        let size = std::mem::size_of::<T>() * size;
        ascend_sys::core::aclrtMemcpyAsync(
            dst.as_ptr() as *mut c_void,
            size,
            src as *const c_void,
            size,
            ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_HOST_TO_DEVICE,
            stream.to_raw(),
        )
        .to_result()
    }
}

/// Asynchronously copy memory from device to host on the given stream.
///
/// The copy is enqueued on the stream and will execute asynchronously.
/// The host memory (`dst`) must not be accessed until the copy completes.
///
/// For best performance, `dst` should be page-locked memory allocated via
/// [`malloc_locked`].
///
/// # Safety
///
/// The device allocation must have been initialized and have `size` elements.
/// The host allocation must have space for `size` elements.
/// The host memory must not be read until the stream operation completes.
pub unsafe fn device_to_host_async<T: DeviceSend>(
    src: DevicePtr<T>,
    dst: *mut T,
    size: usize,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        let size = std::mem::size_of::<T>() * size;
        ascend_sys::core::aclrtMemcpyAsync(
            dst as *mut c_void,
            size,
            src.as_ptr() as *mut c_void,
            size,
            ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_DEVICE_TO_HOST,
            stream.to_raw(),
        )
        .to_result()
    }
}

/// Asynchronously copy memory between two device allocations on the given stream.
///
/// The copy is enqueued on the stream and will execute asynchronously.
///
/// # Safety
///
/// Both allocations must have `size` elements. The source must be initialized.
/// Data interaction between two devices must be enabled via [`Device::enable_peer_access`]
/// if the allocations reside on different devices.
pub unsafe fn device_to_device_async<T: DeviceSend>(
    src: DevicePtr<T>,
    dst: DevicePtr<T>,
    size: usize,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        let size = std::mem::size_of::<T>() * size;
        ascend_sys::core::aclrtMemcpyAsync(
            dst.as_ptr() as *mut c_void,
            size,
            src.as_ptr() as *mut c_void,
            size,
            ascend_sys::core::aclrtMemcpyKind_ACL_MEMCPY_DEVICE_TO_DEVICE,
            stream.to_raw(),
        )
        .to_result()
    }
}

/// Asynchronously set device memory to a value on the given stream.
///
/// Sets `count` bytes starting at `dst` to `value`.
///
/// # Safety
///
/// The device allocation must have at least `count` bytes.
pub unsafe fn memset_async<T: DeviceSend>(
    dst: DevicePtr<T>,
    value: u8,
    count: usize,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        ascend_sys::core::aclrtMemsetAsync(
            dst.as_ptr() as *mut c_void,
            count,
            value as i32,
            count,
            stream.to_raw(),
        )
        .to_result()
    }
}
