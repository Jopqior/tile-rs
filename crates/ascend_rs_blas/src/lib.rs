use std::ffi::c_void;

use ascend_rs::prelude::*;
pub use ascend_sys::blas::*;

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum AclTransType {
    TransN = aclTransType_ACL_TRANS_N,
    TransT = aclTransType_ACL_TRANS_T,
    TransNZ = aclTransType_ACL_TRANS_NZ,
    TransNZT = aclTransType_ACL_TRANS_NZ_T,
}

impl From<AclTransType> for u32 {
    fn from(value: AclTransType) -> u32 {
        value as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum AclComputeType {
    HighPrecision = aclComputeType_ACL_COMPUTE_HIGH_PRECISION,
    LowPrecision = aclComputeType_ACL_COMPUTE_LOW_PRECISION,
}

impl From<AclComputeType> for u32 {
    fn from(value: AclComputeType) -> u32 {
        value as u32
    }
}

pub trait ToAclDataType {
    fn get() -> AclDataType;
}

impl ToAclDataType for usize {
    fn get() -> AclDataType {
        AclDataType::Uint32
    }
}

impl ToAclDataType for u8 {
    fn get() -> AclDataType {
        AclDataType::Uint8
    }
}

impl ToAclDataType for u16 {
    fn get() -> AclDataType {
        AclDataType::Uint16
    }
}

impl ToAclDataType for u32 {
    fn get() -> AclDataType {
        AclDataType::Uint32
    }
}

impl ToAclDataType for u64 {
    fn get() -> AclDataType {
        AclDataType::Uint64
    }
}

impl ToAclDataType for i8 {
    fn get() -> AclDataType {
        AclDataType::Int8
    }
}

impl ToAclDataType for i16 {
    fn get() -> AclDataType {
        AclDataType::Int16
    }
}

impl ToAclDataType for i32 {
    fn get() -> AclDataType {
        AclDataType::Int32
    }
}

impl ToAclDataType for i64 {
    fn get() -> AclDataType {
        AclDataType::Int64
    }
}

impl ToAclDataType for f32 {
    fn get() -> AclDataType {
        AclDataType::Float
    }
}

impl ToAclDataType for f64 {
    fn get() -> AclDataType {
        AclDataType::Double
    }
}

impl ToAclDataType for AclFloat16 {
    fn get() -> AclDataType {
        AclDataType::Float16
    }
}

// ---- GEMM (matrix-matrix multiply): C = alpha * A * B + beta * C ----

/// Generic GEMM with configurable data types (async on stream).
///
/// Computes `C = alpha * op(A) * op(B) + beta * C` where `op()` is
/// controlled by the transpose parameters.
///
/// # Safety
///
/// The device buffers must contain valid data with correct dimensions.
/// The operation executes asynchronously on the given stream.
pub unsafe fn acl_blas_gemm_ex<T: DeviceSend + ToAclDataType>(
    trans_a: AclTransType,
    trans_b: AclTransType,
    trans_c: AclTransType,
    m: i32,
    n: i32,
    k: i32,
    alpha: &DeviceBox<T>,
    matrix_a: &DeviceBuffer<T>,
    lda: i32,
    matrix_b: &DeviceBuffer<T>,
    ldb: i32,
    beta: &DeviceBox<T>,
    matrix_c: &mut DeviceBuffer<T>,
    ldc: i32,
    compute_type: AclComputeType,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        ascend_sys::blas::aclblasGemmEx(
            trans_a.into(),
            trans_b.into(),
            trans_c.into(),
            m,
            n,
            k,
            alpha.as_ptr() as *const c_void,
            matrix_a.as_ptr() as *const c_void,
            lda,
            <T as ToAclDataType>::get().into(),
            matrix_b.as_ptr() as *const c_void,
            ldb,
            <T as ToAclDataType>::get().into(),
            beta.as_ptr() as *const c_void,
            matrix_c.as_mut_ptr() as *mut c_void,
            ldc,
            <T as ToAclDataType>::get().into(),
            compute_type.into(),
            stream.to_raw(),
        )
        .to_result()
    }
}

/// Half-precision (Float16) GEMM (async on stream).
///
/// Computes `C = alpha * op(A) * op(B) + beta * C` using `AclFloat16` throughout.
///
/// # Safety
///
/// The device buffers must contain valid Float16 data with correct dimensions.
pub unsafe fn acl_blas_hgemm(
    trans_a: AclTransType,
    trans_b: AclTransType,
    trans_c: AclTransType,
    m: i32,
    n: i32,
    k: i32,
    alpha: &DeviceBox<AclFloat16>,
    matrix_a: &DeviceBuffer<AclFloat16>,
    lda: i32,
    matrix_b: &DeviceBuffer<AclFloat16>,
    ldb: i32,
    beta: &DeviceBox<AclFloat16>,
    matrix_c: &mut DeviceBuffer<AclFloat16>,
    ldc: i32,
    compute_type: AclComputeType,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        ascend_sys::blas::aclblasHgemm(
            trans_a.into(),
            trans_b.into(),
            trans_c.into(),
            m,
            n,
            k,
            alpha.as_ptr() as *const u16,
            matrix_a.as_ptr() as *const u16,
            lda,
            matrix_b.as_ptr() as *const u16,
            ldb,
            beta.as_ptr() as *const u16,
            matrix_c.as_mut_ptr() as *mut u16,
            ldc,
            compute_type.into(),
            stream.to_raw(),
        )
        .to_result()
    }
}

/// Int8-input / Int32-output GEMM (async on stream).
///
/// Computes `C = alpha * op(A) * op(B) + beta * C` where A and B are `i8`
/// and alpha, beta, and C are `i32`.
///
/// # Safety
///
/// The device buffers must contain valid data with correct dimensions.
pub unsafe fn acl_blas_s8gemm(
    trans_a: AclTransType,
    trans_b: AclTransType,
    trans_c: AclTransType,
    m: i32,
    n: i32,
    k: i32,
    alpha: &DeviceBox<i32>,
    matrix_a: &DeviceBuffer<i8>,
    lda: i32,
    matrix_b: &DeviceBuffer<i8>,
    ldb: i32,
    beta: &DeviceBox<i32>,
    matrix_c: &mut DeviceBuffer<i32>,
    ldc: i32,
    compute_type: AclComputeType,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        ascend_sys::blas::aclblasS8gemm(
            trans_a.into(),
            trans_b.into(),
            trans_c.into(),
            m,
            n,
            k,
            alpha.as_ptr(),
            matrix_a.as_ptr(),
            lda,
            matrix_b.as_ptr(),
            ldb,
            beta.as_ptr(),
            matrix_c.as_mut_ptr(),
            ldc,
            compute_type.into(),
            stream.to_raw(),
        )
        .to_result()
    }
}

// ---- GEMV (matrix-vector multiply): y = alpha * op(A) * x + beta * y ----

/// Generic GEMV with configurable data types (async on stream).
///
/// Computes `y = alpha * op(A) * x + beta * y` where `op()` is
/// controlled by the transpose parameter.
///
/// # Safety
///
/// The device buffers must contain valid data with correct dimensions.
pub unsafe fn acl_blas_gemv_ex<T: DeviceSend + ToAclDataType>(
    trans_a: AclTransType,
    m: i32,
    n: i32,
    alpha: &DeviceBox<T>,
    a: &DeviceBuffer<T>,
    lda: i32,
    x: &DeviceBuffer<T>,
    incx: i32,
    beta: &DeviceBox<T>,
    y: &mut DeviceBuffer<T>,
    incy: i32,
    compute_type: AclComputeType,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        ascend_sys::blas::aclblasGemvEx(
            trans_a.into(),
            m,
            n,
            alpha.as_ptr() as *const c_void,
            a.as_ptr() as *const c_void,
            lda,
            <T as ToAclDataType>::get().into(),
            x.as_ptr() as *const c_void,
            incx,
            <T as ToAclDataType>::get().into(),
            beta.as_ptr() as *const c_void,
            y.as_mut_ptr() as *mut c_void,
            incy,
            <T as ToAclDataType>::get().into(),
            compute_type.into(),
            stream.to_raw(),
        )
        .to_result()
    }
}

/// Half-precision (Float16) GEMV (async on stream).
///
/// Computes `y = alpha * op(A) * x + beta * y` using `AclFloat16` throughout.
///
/// # Safety
///
/// The device buffers must contain valid Float16 data with correct dimensions.
pub unsafe fn acl_blas_hgemv(
    trans_a: AclTransType,
    m: i32,
    n: i32,
    alpha: &DeviceBox<AclFloat16>,
    a: &DeviceBuffer<AclFloat16>,
    lda: i32,
    x: &DeviceBuffer<AclFloat16>,
    incx: i32,
    beta: &DeviceBox<AclFloat16>,
    y: &mut DeviceBuffer<AclFloat16>,
    incy: i32,
    compute_type: AclComputeType,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        ascend_sys::blas::aclblasHgemv(
            trans_a.into(),
            m,
            n,
            alpha.as_ptr() as *const u16,
            a.as_ptr() as *const u16,
            lda,
            x.as_ptr() as *const u16,
            incx,
            beta.as_ptr() as *const u16,
            y.as_mut_ptr() as *mut u16,
            incy,
            compute_type.into(),
            stream.to_raw(),
        )
        .to_result()
    }
}

/// Int8-input / Int32-output GEMV (async on stream).
///
/// Computes `y = alpha * op(A) * x + beta * y` where A and x are `i8`
/// and alpha, beta, and y are `i32`.
///
/// # Safety
///
/// The device buffers must contain valid data with correct dimensions.
pub unsafe fn acl_blas_s8gemv(
    trans_a: AclTransType,
    m: i32,
    n: i32,
    alpha: &DeviceBox<i32>,
    a: &DeviceBuffer<i8>,
    lda: i32,
    x: &DeviceBuffer<i8>,
    incx: i32,
    beta: &DeviceBox<i32>,
    y: &mut DeviceBuffer<i32>,
    incy: i32,
    compute_type: AclComputeType,
    stream: &AclStream,
) -> AclResult<()> {
    unsafe {
        ascend_sys::blas::aclblasS8gemv(
            trans_a.into(),
            m,
            n,
            alpha.as_ptr(),
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            beta.as_ptr(),
            y.as_mut_ptr(),
            incy,
            compute_type.into(),
            stream.to_raw(),
        )
        .to_result()
    }
}
