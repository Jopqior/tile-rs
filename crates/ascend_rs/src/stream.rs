//! A stream is the execution flow on a device, where tasks within the same
//! stream are executed in a strictly ordered manner.
//!
//! There are two types of streams: default and explicit.
//!
//! # Default stream
//!
//! When the [`Device`] or [`AclContext`] are created, the system will
//! automatically and implicitly create a default stream. One device corresponds
//! to one default stream. The default Stream cannot be released through the
//! [`Drop`] interface, but is released together with the corresponding resources.
//!
//! # Explicit stream (Recommended)
//!
//! In a process or thread, stream can be explicitly created by calling a [`AclStream`]
//! ctor.
//!
//! # Relationship between threads and contexts
//!
//! Multiple streams can be created within a single thread, and computational tasks on
//! different streams can be executed in parallel. The tasks within each stream are
//! executed in the order they are issued by the stream.

use std::marker::PhantomData;
use std::ptr;

use ascend_sys::core::aclrtStream as RawStream;

use crate::context::AclContext;
use crate::errors::{AclResult, ToAclResult};

/// Handle for the Ascend stream.
///
/// See the module-level documentation for more details.
#[derive(Clone)]
#[repr(transparent)]
pub struct AclStream<'c> {
    stream: RawStream,
    _context: PhantomData<&'c AclContext<'c>>,
}

impl<'c> AclStream<'c> {
    /// Create [`AclStream`].
    pub fn new(_context: &'c AclContext<'c>) -> AclResult<AclStream<'c>> {
        unsafe {
            let mut stream: ascend_sys::core::aclrtStream = ptr::null_mut();
            ascend_sys::core::aclrtCreateStream(&mut stream).to_result()?;
            Ok(AclStream::from_raw(stream))
        }
    }

    /// Creates a stream from a raw object.
    ///
    /// # Safety
    ///
    /// A raw stream must be valid.
    pub unsafe fn from_raw(stream: RawStream) -> Self {
        Self {
            stream,
            _context: PhantomData::default(),
        }
    }

    /// Converts a stream into a raw object.
    pub fn to_raw(&self) -> RawStream {
        self.stream
    }

    /// Blocks the application from running until all tasks in the specified [`AclStream`]
    /// have completed.
    pub fn synchronize(&self) -> AclResult<()> {
        unsafe { ascend_sys::core::aclrtSynchronizeStream(self.stream).to_result() }
    }
}

impl Drop for AclStream<'_> {
    fn drop(&mut self) {
        unsafe {
            ascend_sys::core::aclrtDestroyStream(self.stream);
        }
    }
}
