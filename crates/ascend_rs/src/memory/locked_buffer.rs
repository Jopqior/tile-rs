//! Fixed-size buffer in page-locked memory.

use super::core;
use crate::errors::AclResult;
use crate::types::DeviceSend;

use std::fmt;

pub struct LockedBuffer<T: DeviceSend> {
    ptr: *mut T,
    size: usize,
}

impl<T: DeviceSend + Copy> LockedBuffer<T> {
    /// Allocate a new page-locked buffer and copy slice content
    /// to it.
    pub fn from_slice(slice: &[T]) -> AclResult<Self> {
        unsafe {
            let dst_ptr = core::malloc_locked(slice.len())?;
            core::host_to_host(slice.as_ptr(), dst_ptr, slice.len())?;
            Ok(LockedBuffer::from_raw_parts(dst_ptr, slice.len()))
        }
    }
}

impl<T: DeviceSend> LockedBuffer<T> {
    /// Returns a raw mutable pointer to the page-locked memory buffer.
    ///
    /// The caller must ensure that the buffer outlives the pointer this
    /// function returns, or else it will end up dangling.
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns a raw pointer to the page-locked memory buffer.
    ///
    /// The caller must ensure that the buffer outlives the pointer this
    /// function returns, or else it will end up dangling.
    ///
    /// The caller must also ensure that the memory the pointer points to
    /// is never written to.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the number of elements in the buffer.
    #[inline]
    pub const fn len(&self) -> usize {
        self.size
    }

    /// Creates a `LockedBuffer<T>` directly from a pointer and a length.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated using the [`core::malloc_locked`].
    /// - `T` needs to have the same alignment as what `ptr` was allocated with.
    /// - `size` needs to be less than or equal as what `ptr` was allocated with.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, size: usize) -> Self {
        LockedBuffer { ptr, size }
    }

    /// Extracts a slice containing the entire buffer.
    ///
    /// Equivalent to `&s[..]`.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Extracts a mutable slice of the entire buffer.
    ///
    /// Equivalent to `&mut s[..]`.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Allocates space for a `[T]` of `n` elements in page-locked memory.
    pub unsafe fn uninitialized(size: usize) -> AclResult<Self> {
        unsafe {
            let ptr = core::malloc_locked::<T>(size)?;
            Ok(LockedBuffer::from_raw_parts(ptr, size))
        }
    }
}

impl<T: DeviceSend> std::ops::Deref for LockedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: DeviceSend> std::ops::DerefMut for LockedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: DeviceSend> std::ops::Index<usize> for LockedBuffer<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        // FIXME: how to avoid additional call?
        &self.as_slice()[index]
    }
}

impl<U: DeviceSend, T: PartialEq<U> + DeviceSend> PartialEq<&[U]> for LockedBuffer<T> {
    fn eq(&self, other: &&[U]) -> bool {
        self.as_slice().iter().eq(other.iter())
    }
}

impl<T: fmt::Debug + DeviceSend> fmt::Debug for LockedBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<T: DeviceSend> Drop for LockedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = core::free_locked(self.ptr);
        }
    }
}

#[cfg(test)]
mod test_host_buffer {
    use ::core::assert_eq;

    use super::*;
    use crate::acl::*;
    use crate::device::*;

    #[test]
    fn test_from_slice() {
        let acl = Acl::new().expect("Failed to initialize ACL");
        let _device = Device::new(&acl);

        let slice = [1, 2, 3, 4, 5, 42];
        let host_buff = LockedBuffer::from_slice(&slice).expect("Failed to create LockedBuffer");

        assert_eq!(host_buff, &slice);
        assert_eq!(host_buff[5], slice[5]);
    }
}
