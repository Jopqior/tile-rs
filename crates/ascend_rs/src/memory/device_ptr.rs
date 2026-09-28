use crate::types::DeviceSend;
use std::ptr::NonNull;

/// A pointer to device memory.
#[repr(transparent)]
pub struct DevicePtr<T: DeviceSend> {
    ptr: NonNull<T>,
}

impl<T: DeviceSend> Copy for DevicePtr<T> {}

impl<T: DeviceSend> Clone for DevicePtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: DeviceSend> DevicePtr<T> {
    /// Returns the underlying pointer to the device allocation.
    ///
    /// The caller must ensure that the pointer is not dereferenced
    /// on the host side.
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns the underlying mutable pointer to the device allocation.
    ///
    /// The caller must ensure that the pointer is not dereferenced
    /// on the host side.
    #[must_use]
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.ptr.as_mut() }
    }

    /// Creates a new `DevicePtr`.
    ///
    /// # Safety
    ///
    /// - `ptr` must be non-null.
    /// - `ptr` must have been allocated using the [`core::malloc_device`].
    #[must_use = "missing the pointer can lead to memory leak"]
    #[inline]
    pub unsafe fn new_unchecked(ptr: *mut T) -> Self {
        unsafe {
            DevicePtr {
                ptr: NonNull::new_unchecked(ptr),
            }
        }
    }

    /// Creates a new `DevicePtr`.
    #[inline]
    pub(crate) fn new(ptr: *mut T) -> Option<Self> {
        // FIXME: impl doesn't satisfy `malloc_device` restriction. Should this
        // method also be unsafe?
        if !ptr.is_null() {
            // SAFETY: The pointer is already checked and is not null
            Some(unsafe { Self::new_unchecked(ptr) })
        } else {
            None
        }
    }
}
