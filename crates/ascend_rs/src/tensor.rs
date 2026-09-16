//! Tensor descriptor management for model inference.
//!
//! A tensor descriptor defines the shape, data type, and format of a tensor.
//! Tensor descriptors are used when setting up model inputs and outputs.

use std::ffi::c_void;

use crate::errors::{AclResult, ToAclResult};
use crate::types::AclDataType;

/// Tensor data format.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AclFormat {
    /// NCHW format (batch, channels, height, width).
    Nchw = 0,
    /// NHWC format (batch, height, width, channels).
    Nhwc = 1,
    /// ND format (N-dimensional, no specific layout).
    Nd = 2,
    /// NC1HWC0 format (Ascend internal format).
    Nc1hwc0 = 3,
    /// FRACTAL_Z format (Ascend internal format).
    FractalZ = 4,
    /// FRACTAL_NZ format (Ascend internal format).
    FractalNz = 29,
    /// Undefined format.
    Undefined = 27,
}

/// RAII wrapper for an ACL tensor descriptor.
///
/// Describes the shape, data type, and format of a tensor. Used when configuring
/// model inputs/outputs and operator execution.
pub struct TensorDesc {
    desc: *mut c_void,
}

impl TensorDesc {
    /// Create a new tensor descriptor.
    ///
    /// # Arguments
    ///
    /// * `shape` - The dimensions of the tensor (e.g., `&[1, 3, 224, 224]` for NCHW)
    /// * `data_type` - The element data type
    /// * `format` - The data layout format
    pub fn new(shape: &[i64], data_type: AclDataType, format: AclFormat) -> AclResult<Self> {
        unsafe {
            let desc = ascend_sys::core::aclCreateTensorDesc(
                data_type as i32,
                shape.len() as i32,
                shape.as_ptr(),
                format as i32,
            ) as *mut c_void;
            if desc.is_null() {
                return Err(crate::errors::AclError::Inner(
                    crate::errors::AclInnerError::AclInnerErrorInvalidParam,
                ));
            }
            Ok(Self { desc })
        }
    }

    /// Create a scalar tensor descriptor (0-dimensional).
    pub fn scalar(data_type: AclDataType) -> AclResult<Self> {
        Self::new(&[], data_type, AclFormat::Nd)
    }

    /// Returns the raw descriptor pointer.
    pub fn as_ptr(&self) -> *const c_void {
        self.desc
    }

    /// Returns the raw mutable descriptor pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.desc
    }

    /// Get the number of dimensions.
    pub fn num_dims(&self) -> usize {
        unsafe { ascend_sys::core::aclGetTensorDescNumDims(self.desc as *const _) as usize }
    }

    /// Get the size of a specific dimension.
    pub fn dim_size(&self, index: usize) -> i64 {
        unsafe {
            let mut size: i64 = 0;
            ascend_sys::core::aclGetTensorDescDimV2(self.desc as *const _, index, &mut size);
            size
        }
    }

    /// Get the shape as a vector.
    pub fn shape(&self) -> Vec<i64> {
        let n = self.num_dims();
        (0..n).map(|i| self.dim_size(i)).collect()
    }

    /// Get the total number of elements.
    pub fn num_elements(&self) -> usize {
        let shape = self.shape();
        if shape.is_empty() {
            1
        } else {
            shape.iter().map(|&d| d as usize).product()
        }
    }

    /// Get the data type.
    pub fn data_type(&self) -> i32 {
        unsafe { ascend_sys::core::aclGetTensorDescType(self.desc as *const _) }
    }

    /// Get the format.
    pub fn format(&self) -> i32 {
        unsafe { ascend_sys::core::aclGetTensorDescFormat(self.desc as *const _) }
    }

    /// Get the byte size of the tensor data.
    pub fn byte_size(&self) -> usize {
        unsafe { ascend_sys::core::aclGetTensorDescSize(self.desc as *const _) }
    }

    /// Set the tensor name.
    pub fn set_name(&mut self, name: &str) {
        let c_name = std::ffi::CString::new(name).expect("invalid tensor name");
        unsafe {
            ascend_sys::core::aclSetTensorDescName(self.desc as *mut _, c_name.as_ptr());
        }
    }
}

impl Drop for TensorDesc {
    fn drop(&mut self) {
        unsafe {
            ascend_sys::core::aclDestroyTensorDesc(self.desc as *const _);
        }
    }
}

/// A data buffer that holds tensor data for model inference.
///
/// This is an RAII wrapper around `aclDataBuffer`. It holds a pointer
/// to device memory and the buffer size.
pub struct DataBuffer {
    buffer: *mut c_void,
}

impl DataBuffer {
    /// Create a data buffer wrapping existing device memory.
    ///
    /// # Arguments
    ///
    /// * `data` - Pointer to device memory
    /// * `size` - Size of the buffer in bytes
    ///
    /// # Safety
    ///
    /// The `data` pointer must point to valid device memory of at least `size` bytes.
    /// The memory must remain valid for the lifetime of this `DataBuffer`.
    pub unsafe fn new(data: *mut c_void, size: usize) -> AclResult<Self> {
        let buffer = unsafe { ascend_sys::core::aclCreateDataBuffer(data, size) as *mut c_void };
        if buffer.is_null() {
            return Err(crate::errors::AclError::Inner(
                crate::errors::AclInnerError::AclInnerErrorRtMemoryAllocation,
            ));
        }
        Ok(Self { buffer })
    }

    /// Returns the raw buffer pointer.
    pub fn as_ptr(&self) -> *const c_void {
        self.buffer
    }

    /// Returns the raw mutable buffer pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.buffer
    }

    /// Get the device memory address held by this buffer.
    pub fn data(&self) -> *mut c_void {
        unsafe { ascend_sys::core::aclGetDataBufferAddr(self.buffer as *const _) }
    }

    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        unsafe { ascend_sys::core::aclGetDataBufferSizeV2(self.buffer as *const _) as usize }
    }
}

impl Drop for DataBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::core::aclDestroyDataBuffer(self.buffer as *mut _);
        }
    }
}

/// A dataset containing multiple data buffers for model inference.
///
/// A dataset groups together the input or output buffers needed for a model
/// execution call.
pub struct Dataset {
    dataset: *mut c_void,
}

impl Dataset {
    /// Create a new empty dataset.
    pub fn new() -> AclResult<Self> {
        unsafe {
            let dataset = ascend_sys::core::aclmdlCreateDataset() as *mut c_void;
            if dataset.is_null() {
                return Err(crate::errors::AclError::Inner(
                    crate::errors::AclInnerError::AclInnerErrorRtMemoryAllocation,
                ));
            }
            Ok(Self { dataset })
        }
    }

    /// Add a data buffer to the dataset.
    ///
    /// Buffers are added in order — the first buffer added corresponds to the
    /// first input/output of the model.
    pub fn add_buffer(&mut self, buffer: &DataBuffer) -> AclResult<()> {
        unsafe {
            ascend_sys::core::aclmdlAddDatasetBuffer(
                self.dataset as *mut _,
                buffer.buffer as *mut _,
            )
            .to_result()
        }
    }

    /// Get the number of buffers in the dataset.
    pub fn num_buffers(&self) -> usize {
        unsafe { ascend_sys::core::aclmdlGetDatasetNumBuffers(self.dataset as *const _) as usize }
    }

    /// Returns the raw dataset pointer.
    pub fn as_ptr(&self) -> *const c_void {
        self.dataset
    }

    /// Returns the raw mutable dataset pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.dataset
    }
}

impl Drop for Dataset {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::core::aclmdlDestroyDataset(self.dataset as *mut _);
        }
    }
}
