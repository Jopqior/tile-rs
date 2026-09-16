use core::cmp::Ord;
use num_traits::FromPrimitive;
use std::fmt;
use std::mem;

use crate::memory::DevicePtr;

#[repr(i32)]
#[derive(Debug, Clone, Copy, FromPrimitive)]
pub enum AclDataType {
    Undefined = ascend_sys::core::aclDataType_ACL_DT_UNDEFINED,
    Float = ascend_sys::core::aclDataType_ACL_FLOAT,
    Float16 = ascend_sys::core::aclDataType_ACL_FLOAT16,
    Int8 = ascend_sys::core::aclDataType_ACL_INT8,
    Int32 = ascend_sys::core::aclDataType_ACL_INT32,
    Uint8 = ascend_sys::core::aclDataType_ACL_UINT8,
    Int16 = ascend_sys::core::aclDataType_ACL_INT16,
    Uint16 = ascend_sys::core::aclDataType_ACL_UINT16,
    Uint32 = ascend_sys::core::aclDataType_ACL_UINT32,
    Int64 = ascend_sys::core::aclDataType_ACL_INT64,
    Uint64 = ascend_sys::core::aclDataType_ACL_UINT64,
    Double = ascend_sys::core::aclDataType_ACL_DOUBLE,
    Bool = ascend_sys::core::aclDataType_ACL_BOOL,
    String = ascend_sys::core::aclDataType_ACL_STRING,
    Complex64 = ascend_sys::core::aclDataType_ACL_COMPLEX64,
    Complex128 = ascend_sys::core::aclDataType_ACL_COMPLEX128,
    Bf16 = ascend_sys::core::aclDataType_ACL_BF16,
    Int4 = ascend_sys::core::aclDataType_ACL_INT4,
    Uint1 = ascend_sys::core::aclDataType_ACL_UINT1,
    Complex32 = ascend_sys::core::aclDataType_ACL_COMPLEX32,
}

impl From<AclDataType> for i32 {
    fn from(data_type: AclDataType) -> Self {
        data_type as i32
    }
}

impl From<i32> for AclDataType {
    fn from(value: i32) -> Self {
        AclDataType::from_i32(value).unwrap_or(AclDataType::Undefined)
    }
}

impl AclDataType {
    pub fn size_of(&self) -> usize {
        match self {
            AclDataType::Undefined => 0,
            AclDataType::Float => mem::size_of::<f32>(),
            AclDataType::Float16 => mem::size_of::<AclFloat16>(), // Half precision (16-bit) float
            AclDataType::Int8 => mem::size_of::<i8>(),
            AclDataType::Int32 => mem::size_of::<i32>(),
            AclDataType::Uint8 => mem::size_of::<u8>(),
            AclDataType::Int16 => mem::size_of::<i16>(),
            AclDataType::Uint16 => mem::size_of::<u16>(),
            AclDataType::Uint32 => mem::size_of::<u32>(),
            AclDataType::Int64 => mem::size_of::<i64>(),
            AclDataType::Uint64 => mem::size_of::<u64>(),
            AclDataType::Double => mem::size_of::<f64>(),
            AclDataType::Bool => mem::size_of::<bool>(),
            AclDataType::String => todo!(), // FIXME: Is it C-style str?
            AclDataType::Complex64 => 8,    // 64-bit complex number (2x f32)
            AclDataType::Complex128 => 16,  // 128-bit complex number (2x f64)
            AclDataType::Bf16 => 2,         // Bfloat16 (16-bit)
            AclDataType::Int4 => 1,         // Assuming 4-bit integer representation
            AclDataType::Uint1 => 1, // Assuming 1-bit integer representation (stored in a byte)
            AclDataType::Complex32 => 8, // 32-bit complex number (2x f32)
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AclFloat16(u16);

impl From<f32> for AclFloat16 {
    fn from(value: f32) -> Self {
        Self(unsafe { ascend_sys::core::aclFloatToFloat16(value) })
    }
}

impl From<AclFloat16> for f32 {
    fn from(value: AclFloat16) -> f32 {
        unsafe { ascend_sys::core::aclFloat16ToFloat(value.0) }
    }
}

impl fmt::Display for AclFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = f32::from(*self);
        val.fmt(f)
    }
}

impl std::ops::Add for AclFloat16 {
    // FIXME: option + overflow check. And unchecked `+` for f32.
    type Output = AclFloat16;

    fn add(self, rhs: AclFloat16) -> AclFloat16 {
        let lhs: f32 = self.into();
        let rhs: f32 = rhs.into();
        AclFloat16::from(lhs + rhs)
    }
}

// It's not safe to transer certain types(pointers, for example) to NPU.
pub unsafe trait DeviceSend /* FIXME: `: ToAclDataType?` */ {}

macro_rules! impl_device_send {
    {$($t:ty),*} => {
        $(
            unsafe impl DeviceSend for $t {}
        )*
    };
}

impl_device_send! {
    usize, u8, u16, u32, u64,
    isize, i8, i16, i32, i64,
    f32, f64,

    AclFloat16
}

unsafe impl<T: DeviceSend> DeviceSend for DevicePtr<T> {}
