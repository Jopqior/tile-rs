//! A user thread will always be bound to a context, and all device resource usage or
//! scheduling must be managed via the context. There is only one context currently
//! in use within a thread, and the context is already associated with the device
//! that the current thread will use.
//!
//! There are two types of contexts: default and explicit.
//!
//! # Default context
//!
//! When the [`Device`] is created, the system will automatically and implicitly
//! create a default [`AclContext`]. One device corresponds to one default context.
//! The default context cannot be released through the [`Drop`] interface, but is
//! released together with the corresponding device.
//!
//! Implicit Context creation is suitable for simple applications without complex
//! interaction logic, but the drawback is that in multithreaded programming,
//! the execution result depends on the order of thread scheduling.
//!
//! # Explicit context (Recommended)
//!
//! An explicit context will be created by calling [`AclContext`] ctors
//! and it's destruction will be achieved by the [`AclContext::drop`] invocation.
//!
//! An explicitly created context suitable for applications with large and complex
//! interactive logic, and facilitates improved program readability and maintainability.
//!
//! # Relationship between threads and contexts
//!
//! A process can create multiple contexts, but a thread can only use one context
//! at a time. If multiple contexts are created within a thread, the thread will
//! use the last context created by default. One can explicitly specify the current
//! thread's context through [`set_current_context`].

use std::marker::PhantomData;
use std::ptr;

use ascend_sys::core::aclrtContext as RawContext;

use crate::device::Device;
use crate::errors::{AclResult, ToAclResult};

/// Handler for the Ascend context.
///
/// See the module-level documentation for more details.
pub struct AclContext<'d> {
    context: RawContext,
    // Context is bound to ad device.
    _device: PhantomData<&'d Device<'d>>,
}

impl<'d> AclContext<'d> {
    /// Create an explicit [`AclContext`].
    pub fn new(device: &'d Device<'d>) -> AclResult<AclContext<'d>> {
        unsafe {
            let mut context_ptr: ascend_sys::core::aclrtContext = ptr::null_mut();
            ascend_sys::core::aclrtCreateContext(&mut context_ptr, device.to_raw()).to_result()?;
            Ok(AclContext::from_raw(context_ptr))
        }
    }

    /// Creates a context from a raw object.
    ///
    /// # Safety
    ///
    /// A raw context must be valid.
    pub unsafe fn from_raw(context: RawContext) -> Self {
        Self {
            context,
            _device: PhantomData::default(),
        }
    }

    /// Converts a context into a raw object.
    pub fn to_raw(&self) -> RawContext {
        self.context
    }
}

/// Set the thread's context.
///
/// If you explicitly create a context (e.g., ctx1) in a thread (e.g., thread1),
/// you do not need to call this method to specify the context of that thread.
/// The system will use ctx1 as the context of thread1 by default. If a context
/// is created implicitly, the system will use the default context as the thread's
/// context.
pub fn set_current_context<'d>(ctxt: &AclContext<'d>) -> AclResult<()> {
    unsafe {
        ascend_sys::core::aclrtSetCurrentContext(ctxt.context).to_result()?;
        Ok(())
    }
}

// FIXME: not very clear which lifetime parameter have to be here:
// pub fn get_current_context(???) > AclResult<AclContext<???>> { todo!() }

impl Drop for AclContext<'_> {
    fn drop(&mut self) {
        unsafe {
            ascend_sys::core::aclrtDestroyContext(self.context);
        }
    }
}
