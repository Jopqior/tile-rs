use super::core;
use super::device_ptr::DevicePtr;
use crate::errors::AclResult;
use crate::types::DeviceSend;

pub struct DeviceBox<T: DeviceSend> {
    ptr: DevicePtr<T>,
}

impl<T: DeviceSend> DeviceBox<T> {
    /// Allocates memory on the device and then places `val` into it.
    ///
    /// [`core::AclrtMemMallocPolicy::HugeFirst`] is used as a allocation default policy.
    pub fn new(val: &T) -> AclResult<Self> {
        Self::new_with_policy(val, core::AclrtMemMallocPolicy::HugeFirst)
    }

    /// Allocates memory on the device and then places `val` into it.
    pub fn new_with_policy(val: &T, policy: core::AclrtMemMallocPolicy) -> AclResult<Self> {
        unsafe {
            let dst_ptr = core::device_malloc(std::mem::size_of::<T>(), policy)?;
            core::host_to_device(std::ptr::from_ref(val), dst_ptr, 1)?;
            return Ok(DeviceBox { ptr: dst_ptr });
        };
    }

    /// Copy data back from the device to host memory.
    pub fn to_host(&self, val: &mut T) -> AclResult<()> {
        unsafe {
            let dst_ptr = std::ptr::from_ref(val) as *mut T;
            core::device_to_host(self.ptr, dst_ptr, 1)?;
            Ok(())
        }
    }

    /// Returns a raw mutable pointer to the device memory box.
    ///
    /// * The caller must ensure that the box outlives the pointer this
    /// function returns, or else it will end up dangling.
    ///
    /// * The caller must ensure that the pointer is not dereferenced
    /// on the host side.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the device memory box.
    ///
    /// * The caller must ensure that the box outlives the pointer this
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
}

impl<T: DeviceSend> Drop for DeviceBox<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = core::device_free(self.ptr);
        }
    }
}
