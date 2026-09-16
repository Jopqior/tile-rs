use std::ffi::{CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr;

use crate::errors::{AclError, AclResult, ToAclResult};

/// Central data structure of the CANN framework. It's responsible for
/// the runtime configuration, initialization and deinitialization.
pub struct Acl {
    _phantom: PhantomData<()>,
}

impl Acl {
    /// Creates a new ACL instance with the default configuration.
    pub fn new() -> AclResult<Self> {
        unsafe {
            ascend_sys::core::aclInit(ptr::null()).to_result()?;
            Ok(Acl {
                _phantom: PhantomData::default(),
            })
        }
    }

    /// Creates a new ACL instance from a [`Path`] pointing to JSON configuration file.
    /// Configuration parameters can be found here:
    /// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/appdevgapi/aclcppdevg_03_0022.html
    pub fn from_config(config: impl AsRef<Path>) -> AclResult<Self> {
        let config = config.as_ref().to_str().unwrap();
        let cstr_config =
            CString::new(config).expect("path given to Acl::from_path_config is empty");
        Self::from_cstr_config(&cstr_config)
    }

    /// Creates a new ACL instance from a [`CStr`] pointing to JSON configuration file.
    /// Configuration parameters can be found here:
    /// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/appdevgapi/aclcppdevg_03_0022.html
    pub fn from_cstr_config(config_path: &CStr) -> AclResult<Self> {
        unsafe {
            ascend_sys::core::aclInit(config_path.as_ptr()).to_result()?;
            Ok(Acl {
                _phantom: PhantomData::default(),
            })
        }
    }

    /// Obtain the current execution mode of the AI ​​software stack.
    pub fn get_run_mode(&self) -> AclResult<RunMode> {
        unsafe {
            let mut run_mode = 0u32;
            ascend_sys::core::aclrtGetRunMode(&mut run_mode).to_result()?;
            RunMode::try_from(run_mode)
        }
    }

    /// Set the directory from a [`Path`] for loading the model file, which is a *.om file
    /// compiled from a single operator.
    pub fn set_model_dir(&self, model_dir: impl AsRef<Path>) -> AclResult<()> {
        let model_dir = model_dir.as_ref().to_str().unwrap();
        let model_dir = CString::new(model_dir).expect("path given to Acl::set_model_dir is empty");
        self.set_model_dir_from_cstr(&model_dir)
    }

    /// Set the directory from a [`CStr`] for loading the model file, which is a *.om file
    /// compiled from a single operator.
    pub fn set_model_dir_from_cstr(&self, cstr: &CStr) -> AclResult<()> {
        unsafe {
            ascend_sys::core::aclopSetModelDir(cstr.as_ptr()).to_result()?;
            Ok(())
        }
    }
}

/// Represents the running mode of the AI software stack.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    /// AI software stack running on the Device's Control CPU or board-side environment.
    ACLDevice = 0,

    /// AI software stack running on the Host CPU.
    ACLHost = 1,
}

impl fmt::Display for RunMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RunMode::ACLDevice => {
                write!(
                    f,
                    "AI software stack running on Device's Control CPU or board-side environment. Not supported by Atlas training series products."
                )
            }
            RunMode::ACLHost => {
                write!(f, "AI software stack running on Host CPU.")
            }
        }
    }
}

impl TryFrom<u32> for RunMode {
    type Error = AclError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(RunMode::ACLDevice),
            1 => Ok(RunMode::ACLHost),
            other => Err(AclError::CantCreateRunMode(other)),
        }
    }
}

impl From<RunMode> for u32 {
    fn from(value: RunMode) -> u32 {
        match value {
            RunMode::ACLDevice => 0,
            RunMode::ACLHost => 1,
        }
    }
}

impl Drop for Acl {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::core::aclFinalize();
        }
    }
}
