use super::core;
use super::device_ptr::DevicePtr;
use super::locked_buffer::LockedBuffer;

use crate::errors::AclResult;
use crate::types::DeviceSend;

pub struct DeviceBuffer<T: DeviceSend> {
    ptr: DevicePtr<T>,
    size: usize,
}

impl<T: DeviceSend + Copy> DeviceBuffer<T> {
    /// Allocate a new device buffer and copy slice content to it.
    ///
    /// [`core::AclrtMemMallocPolicy::HugeFirst`] is used as a allocation default policy.
    #[inline]
    pub fn from_slice(slice: &[T]) -> AclResult<Self> {
        Self::from_slice_with_policy(slice, core::AclrtMemMallocPolicy::HugeFirst)
    }

    /// Allocate a new device buffer with specified [`core::AclrtMemMallocPolicy`] and
    /// copy slice content to it.
    pub fn from_slice_with_policy(
        slice: &[T],
        policy: core::AclrtMemMallocPolicy,
    ) -> AclResult<Self> {
        unsafe {
            let dst_ptr = core::device_malloc(slice.len(), policy)?;
            core::host_to_device(slice.as_ptr(), dst_ptr, slice.len())?;
            Ok(DeviceBuffer::from_raw_parts(dst_ptr, slice.len()))
        }
    }
}

impl<T: DeviceSend> DeviceBuffer<T> {
    /// Returns underlying `DevicePtr`.
    pub fn as_device_ptr(&self) -> DevicePtr<T> {
        self.ptr
    }

    /// Returns the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns a raw mutable pointer to the device memory buffer.
    ///
    /// * The caller must ensure that the buffer outlives the pointer this
    /// function returns, or else it will end up dangling.
    ///
    /// * The caller must also ensure that the memory the pointer points to
    /// is never written to.
    ///
    /// * The caller must ensure that the pointer is not dereferenced
    /// on the host side.
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_mut_ptr()
    }

    /// Returns a raw pointer to the device memory buffer.
    ///
    /// * The caller must ensure that the buffer outlives the pointer this
    /// function returns, or else it will end up dangling.
    ///
    /// * The caller must ensure that the pointer is not dereferenced
    /// on the host side.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Creates a `DeviceBuffer<T>` directly from a pointer and a length.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated using the [`core::device_malloc`].
    /// - `T` needs to have the same alignment as what `ptr` was allocated with.
    /// - `size` needs to be less than or equal as what `ptr` was allocated with.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: DevicePtr<T>, size: usize) -> Self {
        DeviceBuffer { ptr, size }
    }

    /// Allocates space for a `[T]` of `n` elements in device memory.
    pub unsafe fn uninitialized_with_policy(
        size: usize,
        policy: core::AclrtMemMallocPolicy,
    ) -> AclResult<Self> {
        unsafe {
            let ptr = core::device_malloc(size * std::mem::size_of::<T>(), policy)?;
            Ok(DeviceBuffer::from_raw_parts(ptr, size))
        }
    }

    /// Allocates space for a `[T]` of `n` elements in device memory.

    /// [`core::AclrtMemMallocPolicy::HugeFirst`] is used as a default allocation policy.
    #[inline]
    pub unsafe fn uninitialized(size: usize) -> AclResult<Self> {
        unsafe { Self::uninitialized_with_policy(size, core::AclrtMemMallocPolicy::HugeFirst) }
    }

    /// Copy data back from the device to page-locked memory.
    pub fn to_host(&self) -> AclResult<LockedBuffer<T>> {
        unsafe {
            let dst_ptr = core::malloc_locked(self.size)?;
            core::device_to_host(self.ptr, dst_ptr, self.size)?;
            Ok(LockedBuffer::from_raw_parts(dst_ptr, self.size))
        }
    }

    /// Extracts a slice containing the entire buffer.
    ///
    /// Equivalent to `&s[..]`.
    ///
    /// The caller must ensure that the data is not accessed on the host side.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Extracts a mutable slice of the entire buffer.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    /// The caller must ensure that the data is not accessed on the host side.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

impl<T: DeviceSend> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = core::device_free(self.ptr);
        }
    }
}
