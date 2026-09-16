//! Event management for synchronization and timing.
//!
//! Events are used for synchronization between streams and for measuring
//! elapsed time between operations on the device.
//!
//! # Example
//!
//! ```no_run
//! use ascend_rs::prelude::*;
//!
//! let acl = Acl::new().unwrap();
//! let device = Device::new(&acl).unwrap();
//! let context = AclContext::new(&device).unwrap();
//! let stream = AclStream::new(&context).unwrap();
//!
//! let start = AclEvent::new().unwrap();
//! let end = AclEvent::new().unwrap();
//!
//! start.record(&stream).unwrap();
//! // ... launch kernels or memory operations on stream ...
//! end.record(&stream).unwrap();
//!
//! stream.synchronize().unwrap();
//! let elapsed_ms = AclEvent::elapsed_time(&start, &end).unwrap();
//! ```

use std::ptr;

use crate::errors::{AclResult, ToAclResult};
use crate::stream::AclStream;

/// RAII wrapper for an ACL event.
///
/// Events are used for two purposes:
/// 1. **Synchronization**: A stream can wait on an event recorded in another stream,
///    enabling cross-stream dependencies.
/// 2. **Timing**: Measuring elapsed time between two recorded events.
///
/// The event is automatically destroyed when this value is dropped.
pub struct AclEvent {
    event: ascend_sys::core::aclrtEvent,
}

impl AclEvent {
    /// Create a new event.
    pub fn new() -> AclResult<Self> {
        unsafe {
            let mut event: ascend_sys::core::aclrtEvent = ptr::null_mut();
            ascend_sys::core::aclrtCreateEvent(&mut event).to_result()?;
            Ok(Self { event })
        }
    }

    /// Creates an event from a raw handle.
    ///
    /// # Safety
    ///
    /// The raw event must be valid and not already managed by another `AclEvent`.
    pub unsafe fn from_raw(event: ascend_sys::core::aclrtEvent) -> Self {
        Self { event }
    }

    /// Returns the raw event handle.
    pub fn to_raw(&self) -> ascend_sys::core::aclrtEvent {
        self.event
    }

    /// Record this event on the given stream.
    ///
    /// The event captures the state of the stream at the point of the call.
    /// All operations submitted to the stream before this call will be
    /// completed before the event is considered "recorded".
    pub fn record(&self, stream: &AclStream) -> AclResult<()> {
        unsafe { ascend_sys::core::aclrtRecordEvent(self.event, stream.to_raw()).to_result() }
    }

    /// Block the host thread until this event has been recorded (completed).
    pub fn synchronize(&self) -> AclResult<()> {
        unsafe { ascend_sys::core::aclrtSynchronizeEvent(self.event).to_result() }
    }

    /// Make the given stream wait until this event has been recorded.
    ///
    /// All operations submitted to `stream` after this call will not begin
    /// execution until the event has been recorded.
    pub fn stream_wait(&self, stream: &AclStream) -> AclResult<()> {
        unsafe { ascend_sys::core::aclrtStreamWaitEvent(stream.to_raw(), self.event).to_result() }
    }

    /// Compute the elapsed time in milliseconds between two events.
    ///
    /// Both events must have been recorded and completed before calling this function.
    /// Typically you should call `stream.synchronize()` or `end.synchronize()` first.
    pub fn elapsed_time(start: &AclEvent, end: &AclEvent) -> AclResult<f32> {
        unsafe {
            let mut ms: f32 = 0.0;
            ascend_sys::core::aclrtEventElapsedTime(&mut ms, start.event, end.event).to_result()?;
            Ok(ms)
        }
    }

    /// Query whether this event has completed.
    ///
    /// Returns `Ok(true)` if the event has been recorded and all prior work
    /// has completed, `Ok(false)` if the event is still pending.
    pub fn query(&self) -> AclResult<bool> {
        unsafe {
            let mut event_status: u32 = 0;
            let ret = ascend_sys::core::aclrtQueryEvent(self.event, &mut event_status);
            if ret == 0 {
                Ok(true)
            } else {
                // ACL_ERROR_RT_EVENT_NOT_COMPLETE or similar
                Ok(false)
            }
        }
    }
}

impl Drop for AclEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::core::aclrtDestroyEvent(self.event);
        }
    }
}
