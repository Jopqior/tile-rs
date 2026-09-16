//! This crate provides Rust bindings for CANN framework which allows one to leverage the
//! hardware computing resources of the Ascend AI processor to perform:
//! - deep learning inference computations
//! - image preprocessing
//! - single-operator accelerated computations
//!
//! # Basic Concepts
//!
//! ## Host
//!
//! `Host` refers to the x86 or ARM server.
//!
//! ## Device (or NPU)
//!
//! `Device` refers to the hardware device that has the Ascend AI processor.
//!
//! ## Context
//!
//! `Context` manages the lifecycle of all objects (including `Streams`, `Events`, device memory, etc.).
//! Objects in different `Contexts` are completely isolated. For example, `Streams` and `Events`
//! in different `Contexts` cannot be synchronized.
//!
//! ## Stream
//!
//! `Streams` are used to maintain the execution order of asynchronous operations, ensuring
//! that tasks within the same `Stream` are executed on the `Device` in the order they are
//! called in the application.
//!
//! ## Kernel
//!
//! The actual task to be executed on the `Device`.

#![allow(non_local_definitions)]
#![deny(unused_unsafe)]
#![deny(warnings)]

#[macro_use]
extern crate num_derive;

pub mod acl;
#[cfg(feature = "async_op")]
pub mod async_op;
pub mod context;
pub mod device;
#[cfg(feature = "dvpp")]
pub mod dvpp;
pub mod errors;
pub mod event;
#[cfg(feature = "hccl")]
pub mod hccl;
pub mod kernel;
pub mod memory;
pub mod model;
#[cfg(feature = "profiler")]
pub mod profiling;
pub mod stream;
pub mod tensor;
pub mod types;

/// This module re-exports a number of commonly-used items.
pub mod prelude {
    pub use crate::acl::*;
    pub use crate::context::*;
    pub use crate::device::*;
    pub use crate::errors::*;
    pub use crate::event::*;
    pub use crate::kernel::*;
    pub use crate::memory::*;
    pub use crate::model::*;
    pub use crate::stream::*;
    pub use crate::tensor::*;
    pub use crate::types::*;
}

// FIXME: Do not reexport `ascend_sys` dep. Not yet possible since not all
// API calls are supported.
pub mod ascend_sys {
    pub mod core {
        pub use ascend_sys::core::*;
    }
    #[cfg(feature = "blas")]
    pub mod blas {
        pub use ascend_sys::blas::*;
    }
    #[cfg(feature = "dvpp")]
    pub mod dvpp {
        pub use ascend_sys::dvpp::*;
    }
    #[cfg(feature = "hccl")]
    pub mod hccl {
        pub use ascend_sys::hccl::*;
    }
    #[cfg(feature = "profiler")]
    pub mod prof {
        pub use ascend_sys::prof::*;
    }
}
