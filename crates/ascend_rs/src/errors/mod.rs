use std::fmt;

mod inner;
pub use inner::*;

#[derive(Debug)]
pub enum AclError {
    Inner(AclInnerError),
    CantCreateRunMode(u32),
}

impl fmt::Display for AclError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AclError::Inner(code) => write!(f, "{}", code),
            AclError::CantCreateRunMode(mode) => {
                write!(f, "Failed to recognize `{}` run mode", mode)
            }
        }
    }
}

pub type AclResult<T> = Result<T, AclError>;

impl std::error::Error for AclError {}

impl From<AclInnerError> for AclError {
    fn from(value: AclInnerError) -> Self {
        AclError::Inner(value)
    }
}

pub trait ToAclResult {
    fn to_result(self) -> AclResult<()>;
}

impl ToAclResult for i32 {
    fn to_result(self) -> AclResult<()> {
        let acl_error = AclInnerError::from(self);
        match acl_error {
            AclInnerError::AclSuccess => Ok(()),
            _ => Err(acl_error.into()),
        }
    }
}
