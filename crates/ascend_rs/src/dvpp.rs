//! DVPP (Digital Vision Pre-Processing) for image/video processing on Ascend NPU.
//!
//! DVPP provides hardware-accelerated image preprocessing operations including
//! resize, crop, color conversion, and JPEG/PNG decode/encode. All VPC operations
//! are asynchronous and execute on a stream.
//!
//! # Example
//!
//! ```no_run
//! use ascend_rs::prelude::*;
//! use ascend_rs::dvpp::*;
//!
//! let acl = Acl::new().unwrap();
//! let device = Device::new(&acl).unwrap();
//! let context = AclContext::new(&device).unwrap();
//! let stream = AclStream::new(&context).unwrap();
//!
//! // Create a DVPP channel for VPC operations
//! let mut channel = DvppChannel::new(DvppChannelMode::Vpc).unwrap();
//!
//! // Set up input/output picture descriptors and perform operations...
//! ```

use std::ffi::c_void;
use std::ptr;

use crate::errors::{AclError, AclInnerError, AclResult, ToAclResult};
use crate::stream::AclStream;

// Re-export pixel formats as a Rust enum
/// Pixel format for DVPP image processing.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Yuv400 = ascend_sys::dvpp::PIXEL_FORMAT_YUV_400,
    YuvSemiPlanar420 = ascend_sys::dvpp::PIXEL_FORMAT_YUV_SEMIPLANAR_420,
    YvuSemiPlanar420 = ascend_sys::dvpp::PIXEL_FORMAT_YVU_SEMIPLANAR_420,
    YuvSemiPlanar422 = ascend_sys::dvpp::PIXEL_FORMAT_YUV_SEMIPLANAR_422,
    YvuSemiPlanar422 = ascend_sys::dvpp::PIXEL_FORMAT_YVU_SEMIPLANAR_422,
    YuvSemiPlanar444 = ascend_sys::dvpp::PIXEL_FORMAT_YUV_SEMIPLANAR_444,
    YvuSemiPlanar444 = ascend_sys::dvpp::PIXEL_FORMAT_YVU_SEMIPLANAR_444,
    YuyvPacked422 = ascend_sys::dvpp::PIXEL_FORMAT_YUYV_PACKED_422,
    UyvyPacked422 = ascend_sys::dvpp::PIXEL_FORMAT_UYVY_PACKED_422,
    YvuPlanar420 = ascend_sys::dvpp::PIXEL_FORMAT_YVU_PLANAR_420,
    YuvPacked444 = ascend_sys::dvpp::PIXEL_FORMAT_YUV_PACKED_444,
    Rgb888 = ascend_sys::dvpp::PIXEL_FORMAT_RGB_888,
    Bgr888 = ascend_sys::dvpp::PIXEL_FORMAT_BGR_888,
    Argb8888 = ascend_sys::dvpp::PIXEL_FORMAT_ARGB_8888,
    Abgr8888 = ascend_sys::dvpp::PIXEL_FORMAT_ABGR_8888,
    Rgba8888 = ascend_sys::dvpp::PIXEL_FORMAT_RGBA_8888,
    Bgra8888 = ascend_sys::dvpp::PIXEL_FORMAT_BGRA_8888,
    Bgr888Planar = ascend_sys::dvpp::PIXEL_FORMAT_BGR_888_PLANAR,
    Float32 = ascend_sys::dvpp::PIXEL_FORMAT_FLOAT32,
}

/// DVPP channel mode.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DvppChannelMode {
    /// VPC (Video Processing Channel) for image operations.
    Vpc = ascend_sys::dvpp::DVPP_CHNMODE_VPC,
    /// JPEG decode.
    JpegDecode = ascend_sys::dvpp::DVPP_CHNMODE_JPEGD,
    /// JPEG encode.
    JpegEncode = ascend_sys::dvpp::DVPP_CHNMODE_JPEGE,
    /// PNG decode.
    PngDecode = ascend_sys::dvpp::DVPP_CHNMODE_PNGD,
}

/// Allocate DVPP device memory.
///
/// DVPP operations require memory allocated through this function rather than
/// the standard `device_malloc`.
///
/// # Safety
///
/// The returned pointer must be freed with [`dvpp_free`].
pub unsafe fn dvpp_malloc(size: usize) -> AclResult<*mut c_void> {
    let mut ptr: *mut c_void = ptr::null_mut();
    unsafe {
        (ascend_sys::dvpp::acldvppMalloc(&mut ptr, size) as i32).to_result()?;
    }
    Ok(ptr)
}

/// Free DVPP device memory allocated with [`dvpp_malloc`].
///
/// # Safety
///
/// The pointer must have been allocated with [`dvpp_malloc`] and must not be
/// used after this call.
pub unsafe fn dvpp_free(ptr: *mut c_void) -> AclResult<()> {
    unsafe { (ascend_sys::dvpp::acldvppFree(ptr) as i32).to_result() }
}

// ---- Picture Description ----

/// RAII wrapper for a DVPP picture descriptor.
///
/// Describes an image's data pointer, dimensions, format, and stride for
/// DVPP processing operations.
pub struct PicDesc {
    desc: *mut c_void,
}

impl PicDesc {
    /// Create a new empty picture descriptor.
    pub fn new() -> AclResult<Self> {
        let desc = unsafe { ascend_sys::dvpp::acldvppCreatePicDesc() };
        if desc.is_null() {
            return Err(AclError::Inner(
                AclInnerError::AclInnerErrorRtMemoryAllocation,
            ));
        }
        Ok(Self { desc })
    }

    /// Create a picture descriptor with all fields set.
    pub fn with_params(
        data: *mut c_void,
        size: u32,
        width: u32,
        height: u32,
        format: PixelFormat,
        width_stride: u32,
        height_stride: u32,
    ) -> AclResult<Self> {
        let pic = Self::new()?;
        pic.set_data(data)?;
        pic.set_size(size)?;
        pic.set_width(width)?;
        pic.set_height(height)?;
        pic.set_format(format)?;
        pic.set_width_stride(width_stride)?;
        pic.set_height_stride(height_stride)?;
        Ok(pic)
    }

    pub fn set_data(&self, data: *mut c_void) -> AclResult<()> {
        unsafe { ascend_sys::dvpp::acldvppSetPicDescData(self.desc, data).to_result() }
    }

    pub fn set_size(&self, size: u32) -> AclResult<()> {
        unsafe { ascend_sys::dvpp::acldvppSetPicDescSize(self.desc, size).to_result() }
    }

    pub fn set_format(&self, format: PixelFormat) -> AclResult<()> {
        unsafe { ascend_sys::dvpp::acldvppSetPicDescFormat(self.desc, format as u32).to_result() }
    }

    pub fn set_width(&self, width: u32) -> AclResult<()> {
        unsafe { ascend_sys::dvpp::acldvppSetPicDescWidth(self.desc, width).to_result() }
    }

    pub fn set_height(&self, height: u32) -> AclResult<()> {
        unsafe { ascend_sys::dvpp::acldvppSetPicDescHeight(self.desc, height).to_result() }
    }

    pub fn set_width_stride(&self, stride: u32) -> AclResult<()> {
        unsafe { ascend_sys::dvpp::acldvppSetPicDescWidthStride(self.desc, stride).to_result() }
    }

    pub fn set_height_stride(&self, stride: u32) -> AclResult<()> {
        unsafe { ascend_sys::dvpp::acldvppSetPicDescHeightStride(self.desc, stride).to_result() }
    }

    pub fn data(&self) -> *mut c_void {
        unsafe { ascend_sys::dvpp::acldvppGetPicDescData(self.desc) }
    }

    pub fn size(&self) -> u32 {
        unsafe { ascend_sys::dvpp::acldvppGetPicDescSize(self.desc) }
    }

    pub fn format(&self) -> u32 {
        unsafe { ascend_sys::dvpp::acldvppGetPicDescFormat(self.desc) }
    }

    pub fn width(&self) -> u32 {
        unsafe { ascend_sys::dvpp::acldvppGetPicDescWidth(self.desc) }
    }

    pub fn height(&self) -> u32 {
        unsafe { ascend_sys::dvpp::acldvppGetPicDescHeight(self.desc) }
    }

    pub fn width_stride(&self) -> u32 {
        unsafe { ascend_sys::dvpp::acldvppGetPicDescWidthStride(self.desc) }
    }

    pub fn height_stride(&self) -> u32 {
        unsafe { ascend_sys::dvpp::acldvppGetPicDescHeightStride(self.desc) }
    }

    pub fn ret_code(&self) -> u32 {
        unsafe { ascend_sys::dvpp::acldvppGetPicDescRetCode(self.desc) }
    }

    /// Returns the raw descriptor pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.desc
    }
}

impl Drop for PicDesc {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::dvpp::acldvppDestroyPicDesc(self.desc);
        }
    }
}

// ---- ROI Config ----

/// Region of interest configuration for crop and paste operations.
pub struct RoiConfig {
    config: *mut c_void,
}

impl RoiConfig {
    /// Create a new ROI configuration.
    ///
    /// Coordinates are inclusive pixel indices.
    pub fn new(left: u32, right: u32, top: u32, bottom: u32) -> AclResult<Self> {
        let config = unsafe { ascend_sys::dvpp::acldvppCreateRoiConfig(left, right, top, bottom) };
        if config.is_null() {
            return Err(AclError::Inner(
                AclInnerError::AclInnerErrorRtMemoryAllocation,
            ));
        }
        Ok(Self { config })
    }

    /// Update the ROI coordinates.
    pub fn set(&mut self, left: u32, right: u32, top: u32, bottom: u32) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppSetRoiConfig(self.config, left, right, top, bottom).to_result()
        }
    }

    /// Returns the raw config pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.config
    }
}

impl Drop for RoiConfig {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::dvpp::acldvppDestroyRoiConfig(self.config);
        }
    }
}

// ---- Resize Config ----

/// Configuration for image resize operations.
pub struct ResizeConfig {
    config: *mut c_void,
}

impl ResizeConfig {
    /// Create a new resize configuration.
    pub fn new() -> AclResult<Self> {
        let config = unsafe { ascend_sys::dvpp::acldvppCreateResizeConfig() };
        if config.is_null() {
            return Err(AclError::Inner(
                AclInnerError::AclInnerErrorRtMemoryAllocation,
            ));
        }
        Ok(Self { config })
    }

    /// Set the interpolation method.
    pub fn set_interpolation(&mut self, interpolation: u32) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppSetResizeConfigInterpolation(self.config, interpolation)
                .to_result()
        }
    }

    /// Get the interpolation method.
    pub fn interpolation(&self) -> u32 {
        unsafe { ascend_sys::dvpp::acldvppGetResizeConfigInterpolation(self.config) }
    }

    /// Returns the raw config pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.config
    }
}

impl Drop for ResizeConfig {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::dvpp::acldvppDestroyResizeConfig(self.config);
        }
    }
}

// ---- JPEG Encode Config ----

/// Configuration for JPEG encoding quality.
pub struct JpegeConfig {
    config: *mut c_void,
}

impl JpegeConfig {
    /// Create a new JPEG encode configuration.
    pub fn new() -> AclResult<Self> {
        let config = unsafe { ascend_sys::dvpp::acldvppCreateJpegeConfig() };
        if config.is_null() {
            return Err(AclError::Inner(
                AclInnerError::AclInnerErrorRtMemoryAllocation,
            ));
        }
        Ok(Self { config })
    }

    /// Set the JPEG quality level (0-100).
    pub fn set_level(&mut self, level: u32) -> AclResult<()> {
        unsafe { ascend_sys::dvpp::acldvppSetJpegeConfigLevel(self.config, level).to_result() }
    }

    /// Get the JPEG quality level.
    pub fn level(&self) -> u32 {
        unsafe { ascend_sys::dvpp::acldvppGetJpegeConfigLevel(self.config) }
    }

    /// Returns the raw config pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.config
    }
}

impl Drop for JpegeConfig {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::dvpp::acldvppDestroyJpegeConfig(self.config);
        }
    }
}

// ---- DVPP Channel ----

/// RAII wrapper for a DVPP processing channel.
///
/// A channel must be created before performing any DVPP operations.
/// The channel mode determines which operations are available.
pub struct DvppChannel {
    desc: *mut c_void,
}

impl DvppChannel {
    /// Create a new DVPP channel with the specified mode.
    pub fn new(mode: DvppChannelMode) -> AclResult<Self> {
        unsafe {
            let desc = ascend_sys::dvpp::acldvppCreateChannelDesc();
            if desc.is_null() {
                return Err(AclError::Inner(
                    AclInnerError::AclInnerErrorRtMemoryAllocation,
                ));
            }
            ascend_sys::dvpp::acldvppSetChannelDescMode(desc, mode as u32).to_result()?;
            ascend_sys::dvpp::acldvppCreateChannel(desc).to_result()?;
            Ok(Self { desc })
        }
    }

    /// Get the channel ID.
    pub fn channel_id(&self) -> u64 {
        unsafe { ascend_sys::dvpp::acldvppGetChannelDescChannelId(self.desc) }
    }

    /// Asynchronously resize an image.
    pub fn resize_async(
        &mut self,
        input: &mut PicDesc,
        output: &mut PicDesc,
        config: &mut ResizeConfig,
        stream: &AclStream,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppVpcResizeAsync(
                self.desc,
                input.desc,
                output.desc,
                config.config,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Asynchronously crop an image.
    pub fn crop_async(
        &mut self,
        input: &mut PicDesc,
        output: &mut PicDesc,
        crop_area: &mut RoiConfig,
        stream: &AclStream,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppVpcCropAsync(
                self.desc,
                input.desc,
                output.desc,
                crop_area.config,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Asynchronously crop and paste an image.
    pub fn crop_and_paste_async(
        &mut self,
        input: &mut PicDesc,
        output: &mut PicDesc,
        crop_area: &mut RoiConfig,
        paste_area: &mut RoiConfig,
        stream: &AclStream,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppVpcCropAndPasteAsync(
                self.desc,
                input.desc,
                output.desc,
                crop_area.config,
                paste_area.config,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Asynchronously crop and resize an image.
    pub fn crop_resize_async(
        &mut self,
        input: &mut PicDesc,
        output: &mut PicDesc,
        crop_area: &mut RoiConfig,
        resize_config: &mut ResizeConfig,
        stream: &AclStream,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppVpcCropResizeAsync(
                self.desc,
                input.desc,
                output.desc,
                crop_area.config,
                resize_config.config,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Asynchronously convert color space.
    pub fn convert_color_async(
        &mut self,
        input: &mut PicDesc,
        output: &mut PicDesc,
        stream: &AclStream,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppVpcConvertColorAsync(
                self.desc,
                input.desc,
                output.desc,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Asynchronously decode a JPEG image.
    pub fn jpeg_decode_async(
        &mut self,
        data: &[u8],
        output: &mut PicDesc,
        stream: &AclStream,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppJpegDecodeAsync(
                self.desc,
                data.as_ptr() as *const c_void,
                data.len() as u32,
                output.desc,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Asynchronously encode an image to JPEG.
    ///
    /// Returns the actual encoded size.
    pub fn jpeg_encode_async(
        &mut self,
        input: &mut PicDesc,
        output_buf: *mut c_void,
        output_size: &mut u32,
        config: &mut JpegeConfig,
        stream: &AclStream,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppJpegEncodeAsync(
                self.desc,
                input.desc,
                output_buf as *const c_void,
                output_size,
                config.config,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Asynchronously decode a PNG image.
    pub fn png_decode_async(
        &mut self,
        data: &[u8],
        output: &mut PicDesc,
        stream: &AclStream,
    ) -> AclResult<()> {
        unsafe {
            ascend_sys::dvpp::acldvppPngDecodeAsync(
                self.desc,
                data.as_ptr() as *const c_void,
                data.len() as u32,
                output.desc,
                stream.to_raw(),
            )
            .to_result()
        }
    }

    /// Returns the raw channel descriptor pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.desc
    }
}

impl Drop for DvppChannel {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::dvpp::acldvppDestroyChannel(self.desc);
            let _ = ascend_sys::dvpp::acldvppDestroyChannelDesc(self.desc);
        }
    }
}

// ---- Image info queries ----

/// Information about a JPEG image.
#[derive(Debug, Clone)]
pub struct JpegImageInfo {
    pub width: u32,
    pub height: u32,
    pub components: i32,
}

/// Query JPEG image dimensions and component count from encoded data.
pub fn jpeg_get_image_info(data: &[u8]) -> AclResult<JpegImageInfo> {
    unsafe {
        let mut width: u32 = 0;
        let mut height: u32 = 0;
        let mut components: i32 = 0;
        ascend_sys::dvpp::acldvppJpegGetImageInfo(
            data.as_ptr() as *const c_void,
            data.len() as u32,
            &mut width,
            &mut height,
            &mut components,
        )
        .to_result()?;
        Ok(JpegImageInfo {
            width,
            height,
            components,
        })
    }
}

/// Query PNG image dimensions and component count from encoded data.
pub fn png_get_image_info(data: &[u8]) -> AclResult<JpegImageInfo> {
    unsafe {
        let mut width: u32 = 0;
        let mut height: u32 = 0;
        let mut components: i32 = 0;
        ascend_sys::dvpp::acldvppPngGetImageInfo(
            data.as_ptr() as *const c_void,
            data.len() as u32,
            &mut width,
            &mut height,
            &mut components,
        )
        .to_result()?;
        Ok(JpegImageInfo {
            width,
            height,
            components,
        })
    }
}

/// Predict the buffer size needed to decode a JPEG image.
pub fn jpeg_predict_dec_size(data: &[u8], output_format: PixelFormat) -> AclResult<u32> {
    unsafe {
        let mut dec_size: u32 = 0;
        ascend_sys::dvpp::acldvppJpegPredictDecSize(
            data.as_ptr() as *const c_void,
            data.len() as u32,
            output_format as u32,
            &mut dec_size,
        )
        .to_result()?;
        Ok(dec_size)
    }
}

/// Predict the buffer size needed to decode a PNG image.
pub fn png_predict_dec_size(data: &[u8], output_format: PixelFormat) -> AclResult<u32> {
    unsafe {
        let mut dec_size: u32 = 0;
        ascend_sys::dvpp::acldvppPngPredictDecSize(
            data.as_ptr() as *const c_void,
            data.len() as u32,
            output_format as u32,
            &mut dec_size,
        )
        .to_result()?;
        Ok(dec_size)
    }
}
