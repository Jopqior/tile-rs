//! Model loading and inference.
//!
//! This module provides RAII wrappers for loading offline model files (`.om`)
//! and executing inference on the Ascend NPU.
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
//! // Load model
//! let model = Model::from_file("model.om").unwrap();
//!
//! // Query model info
//! let desc = model.description().unwrap();
//! println!("Inputs: {}, Outputs: {}", desc.num_inputs(), desc.num_outputs());
//!
//! // Set up input/output datasets and execute...
//! ```

use std::ffi::{CString, c_void};
use std::path::Path;

use crate::errors::{AclResult, ToAclResult};
use crate::stream::AclStream;
use crate::tensor::Dataset;

/// An ACL model loaded into device memory, ready for inference.
///
/// The model is automatically unloaded when this value is dropped.
pub struct Model {
    model_id: u32,
}

impl Model {
    /// Load an offline model from a file (`.om` format).
    ///
    /// The model file is loaded into device memory and can be used for inference.
    pub fn from_file<P: AsRef<Path>>(path: P) -> AclResult<Self> {
        let path_str = path.as_ref().to_str().expect("invalid model path");
        let c_path = CString::new(path_str).expect("invalid model path");
        unsafe {
            let mut model_id: u32 = 0;
            ascend_sys::core::aclmdlLoadFromFile(c_path.as_ptr(), &mut model_id).to_result()?;
            Ok(Self { model_id })
        }
    }

    /// Load a model from a memory buffer.
    ///
    /// # Safety
    ///
    /// The memory buffer must contain a valid `.om` model and remain valid
    /// for the duration of this call.
    pub unsafe fn from_memory(data: &[u8]) -> AclResult<Self> {
        let mut model_id: u32 = 0;
        unsafe {
            ascend_sys::core::aclmdlLoadFromMem(
                data.as_ptr() as *const c_void,
                data.len(),
                &mut model_id,
            )
            .to_result()?;
        }
        Ok(Self { model_id })
    }

    /// Get the model ID.
    pub fn id(&self) -> u32 {
        self.model_id
    }

    /// Get the model description (input/output info).
    pub fn description(&self) -> AclResult<ModelDescription> {
        unsafe {
            let desc = ascend_sys::core::aclmdlCreateDesc();
            if desc.is_null() {
                return Err(crate::errors::AclError::Inner(
                    crate::errors::AclInnerError::AclInnerErrorRtMemoryAllocation,
                ));
            }
            ascend_sys::core::aclmdlGetDesc(desc, self.model_id).to_result()?;
            Ok(ModelDescription {
                desc: desc as *mut c_void,
            })
        }
    }

    /// Execute synchronous inference.
    ///
    /// # Arguments
    ///
    /// * `input` - Dataset containing the model input buffers
    /// * `output` - Dataset that will receive the model output buffers
    pub fn execute(&self, input: &Dataset, output: &mut Dataset) -> AclResult<()> {
        unsafe {
            ascend_sys::core::aclmdlExecute(
                self.model_id,
                input.as_ptr() as *const _,
                output.as_mut_ptr() as *mut _,
            )
            .to_result()
        }
    }

    /// Execute asynchronous inference on a stream.
    ///
    /// The inference is enqueued on the stream and will execute asynchronously.
    /// Use [`AclStream::synchronize`] or events to wait for completion.
    ///
    /// # Arguments
    ///
    /// * `input` - Dataset containing the model input buffers
    /// * `output` - Dataset that will receive the model output buffers
    /// * `stream` - The stream to execute on
    pub fn execute_async(
        &self,
        input: &Dataset,
        output: &mut Dataset,
        stream: &AclStream,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::core::aclmdlExecuteAsync(
                self.model_id,
                input.as_ptr() as *const _,
                output.as_mut_ptr() as *mut _,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Set the dynamic batch size for inference.
    ///
    /// This is used for models that support dynamic batching, where the batch
    /// size is determined at runtime.
    pub fn set_dynamic_batch_size(
        &self,
        input: &Dataset,
        index: usize,
        batch_size: u64,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::core::aclmdlSetDynamicBatchSize(
                self.model_id,
                input.as_ptr() as *mut _,
                index,
                batch_size,
            )
            .to_result()
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::core::aclmdlUnload(self.model_id);
        }
    }
}

/// Description of a loaded model's inputs and outputs.
///
/// Provides metadata about the model's expected input and output tensors,
/// including their shapes, data types, and sizes.
pub struct ModelDescription {
    desc: *mut c_void,
}

impl ModelDescription {
    /// Get the number of input tensors.
    pub fn num_inputs(&self) -> usize {
        unsafe { ascend_sys::core::aclmdlGetNumInputs(self.desc as *mut _) as usize }
    }

    /// Get the number of output tensors.
    pub fn num_outputs(&self) -> usize {
        unsafe { ascend_sys::core::aclmdlGetNumOutputs(self.desc as *mut _) as usize }
    }

    /// Get the byte size of an input tensor by index.
    pub fn input_size(&self, index: usize) -> usize {
        unsafe { ascend_sys::core::aclmdlGetInputSizeByIndex(self.desc as *mut _, index) }
    }

    /// Get the byte size of an output tensor by index.
    pub fn output_size(&self, index: usize) -> usize {
        unsafe { ascend_sys::core::aclmdlGetOutputSizeByIndex(self.desc as *mut _, index) }
    }

    /// Get the input data type by index.
    pub fn input_data_type(&self, index: usize) -> i32 {
        unsafe { ascend_sys::core::aclmdlGetInputDataType(self.desc as *mut _, index) }
    }

    /// Get the output data type by index.
    pub fn output_data_type(&self, index: usize) -> i32 {
        unsafe { ascend_sys::core::aclmdlGetOutputDataType(self.desc as *mut _, index) }
    }

    /// Get the input format by index.
    pub fn input_format(&self, index: usize) -> i32 {
        unsafe { ascend_sys::core::aclmdlGetInputFormat(self.desc as *mut _, index) }
    }

    /// Get the output format by index.
    pub fn output_format(&self, index: usize) -> i32 {
        unsafe { ascend_sys::core::aclmdlGetOutputFormat(self.desc as *mut _, index) }
    }

    /// Get the number of dimensions of an input tensor.
    pub fn input_num_dims(&self, index: usize) -> usize {
        unsafe {
            let mut dims = ascend_sys::core::aclmdlIODims {
                name: [0; 128],
                dimCount: 0,
                dims: [0i64; 128],
            };
            let _ = ascend_sys::core::aclmdlGetInputDims(self.desc as *mut _, index, &mut dims);
            dims.dimCount as usize
        }
    }

    /// Get the shape of an input tensor.
    pub fn input_shape(&self, index: usize) -> Vec<i64> {
        unsafe {
            let mut dims = ascend_sys::core::aclmdlIODims {
                name: [0; 128],
                dimCount: 0,
                dims: [0i64; 128],
            };
            let _ = ascend_sys::core::aclmdlGetInputDims(self.desc as *mut _, index, &mut dims);
            dims.dims[..dims.dimCount as usize].to_vec()
        }
    }

    /// Get the shape of an output tensor.
    pub fn output_shape(&self, index: usize) -> Vec<i64> {
        unsafe {
            let mut dims = ascend_sys::core::aclmdlIODims {
                name: [0; 128],
                dimCount: 0,
                dims: [0i64; 128],
            };
            let _ = ascend_sys::core::aclmdlGetOutputDims(self.desc as *mut _, index, &mut dims);
            dims.dims[..dims.dimCount as usize].to_vec()
        }
    }

    /// Returns the raw description pointer.
    pub fn as_ptr(&self) -> *const c_void {
        self.desc
    }
}

impl Drop for ModelDescription {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::core::aclmdlDestroyDesc(self.desc as *mut _);
        }
    }
}
