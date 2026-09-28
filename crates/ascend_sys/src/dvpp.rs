//! Manual FFI declarations for CANN DVPP (Digital Vision Pre-Processing) APIs.
//!
//! These declarations match `acl/ops/acl_dvpp.h` from the CANN SDK.
//! They can be replaced with bindgen-generated bindings once the header
//! is added to the wrapper.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::c_void;

// ---- Opaque types ----

pub type acldvppPicDesc = c_void;
pub type acldvppBatchPicDesc = c_void;
pub type acldvppRoiConfig = c_void;
pub type acldvppResizeConfig = c_void;
pub type acldvppBorderConfig = c_void;
pub type acldvppLutMap = c_void;
pub type acldvppChannelDesc = c_void;
pub type acldvppJpegeConfig = c_void;
pub type acldvppStreamDesc = c_void;
pub type acldvppHist = c_void;
pub type aclvdecChannelDesc = c_void;
pub type aclvdecFrameConfig = c_void;
pub type aclvencChannelDesc = c_void;
pub type aclvencFrameConfig = c_void;

// ---- Enums ----

pub type acldvppPixelFormat = u32;
pub const PIXEL_FORMAT_YUV_400: acldvppPixelFormat = 0;
pub const PIXEL_FORMAT_YUV_SEMIPLANAR_420: acldvppPixelFormat = 1;
pub const PIXEL_FORMAT_YVU_SEMIPLANAR_420: acldvppPixelFormat = 2;
pub const PIXEL_FORMAT_YUV_SEMIPLANAR_422: acldvppPixelFormat = 3;
pub const PIXEL_FORMAT_YVU_SEMIPLANAR_422: acldvppPixelFormat = 4;
pub const PIXEL_FORMAT_YUV_SEMIPLANAR_444: acldvppPixelFormat = 5;
pub const PIXEL_FORMAT_YVU_SEMIPLANAR_444: acldvppPixelFormat = 6;
pub const PIXEL_FORMAT_YUYV_PACKED_422: acldvppPixelFormat = 7;
pub const PIXEL_FORMAT_UYVY_PACKED_422: acldvppPixelFormat = 8;
pub const PIXEL_FORMAT_YVYU_PACKED_422: acldvppPixelFormat = 9;
pub const PIXEL_FORMAT_VYUY_PACKED_422: acldvppPixelFormat = 10;
pub const PIXEL_FORMAT_YUV_PACKED_444: acldvppPixelFormat = 11;
pub const PIXEL_FORMAT_RGB_888: acldvppPixelFormat = 12;
pub const PIXEL_FORMAT_BGR_888: acldvppPixelFormat = 13;
pub const PIXEL_FORMAT_ARGB_8888: acldvppPixelFormat = 14;
pub const PIXEL_FORMAT_ABGR_8888: acldvppPixelFormat = 15;
pub const PIXEL_FORMAT_RGBA_8888: acldvppPixelFormat = 16;
pub const PIXEL_FORMAT_BGRA_8888: acldvppPixelFormat = 17;
pub const PIXEL_FORMAT_YVU_PLANAR_420: acldvppPixelFormat = 20;
pub const PIXEL_FORMAT_BGR_888_PLANAR: acldvppPixelFormat = 70;
pub const PIXEL_FORMAT_FLOAT32: acldvppPixelFormat = 1002;
pub const PIXEL_FORMAT_UNKNOWN: acldvppPixelFormat = 10000;

pub type acldvppStreamFormat = u32;
pub const H265_MAIN_LEVEL: acldvppStreamFormat = 0;
pub const H264_BASELINE_LEVEL: acldvppStreamFormat = 1;
pub const H264_MAIN_LEVEL: acldvppStreamFormat = 2;
pub const H264_HIGH_LEVEL: acldvppStreamFormat = 3;

pub type acldvppChannelMode = u32;
pub const DVPP_CHNMODE_VPC: acldvppChannelMode = 1;
pub const DVPP_CHNMODE_JPEGD: acldvppChannelMode = 2;
pub const DVPP_CHNMODE_JPEGE: acldvppChannelMode = 4;
pub const DVPP_CHNMODE_PNGD: acldvppChannelMode = 8;

pub type acldvppBorderType = u32;
pub const BORDER_CONSTANT: acldvppBorderType = 0;
pub const BORDER_REPLICATE: acldvppBorderType = 1;
pub const BORDER_REFLECT: acldvppBorderType = 2;
pub const BORDER_REFLECT_101: acldvppBorderType = 3;

pub type acldvppJpegFormat = i32;
pub const ACL_JPEG_CSS_444: acldvppJpegFormat = 0;
pub const ACL_JPEG_CSS_422: acldvppJpegFormat = 1;
pub const ACL_JPEG_CSS_420: acldvppJpegFormat = 2;
pub const ACL_JPEG_CSS_GRAY: acldvppJpegFormat = 3;
pub const ACL_JPEG_CSS_440: acldvppJpegFormat = 4;
pub const ACL_JPEG_CSS_411: acldvppJpegFormat = 5;
pub const ACL_JPEG_CSS_UNKNOWN: acldvppJpegFormat = 1000;

// aclrtStream is *mut c_void (from core bindings)
pub type aclrtStream = *mut c_void;

// ---- Callback types ----

pub type aclvdecCallback = Option<
    unsafe extern "C" fn(
        input: *mut acldvppStreamDesc,
        output: *mut acldvppPicDesc,
        userData: *mut c_void,
    ),
>;

pub type aclvencCallback = Option<
    unsafe extern "C" fn(
        input: *mut acldvppPicDesc,
        output: *mut acldvppStreamDesc,
        userdata: *mut c_void,
    ),
>;

extern "C" {
    // ---- Memory ----
    pub fn acldvppMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    pub fn acldvppFree(devPtr: *mut c_void) -> i32;

    // ---- Channel ----
    pub fn acldvppCreateChannelDesc() -> *mut acldvppChannelDesc;
    pub fn acldvppDestroyChannelDesc(channelDesc: *mut acldvppChannelDesc) -> i32;
    pub fn acldvppCreateChannel(channelDesc: *mut acldvppChannelDesc) -> i32;
    pub fn acldvppDestroyChannel(channelDesc: *mut acldvppChannelDesc) -> i32;
    pub fn acldvppSetChannelDescMode(channelDesc: *mut acldvppChannelDesc, mode: u32) -> i32;
    pub fn acldvppGetChannelDescChannelId(channelDesc: *const acldvppChannelDesc) -> u64;

    // ---- Picture description ----
    pub fn acldvppCreatePicDesc() -> *mut acldvppPicDesc;
    pub fn acldvppDestroyPicDesc(picDesc: *mut acldvppPicDesc) -> i32;
    pub fn acldvppSetPicDescData(picDesc: *mut acldvppPicDesc, dataDev: *mut c_void) -> i32;
    pub fn acldvppSetPicDescSize(picDesc: *mut acldvppPicDesc, size: u32) -> i32;
    pub fn acldvppSetPicDescFormat(picDesc: *mut acldvppPicDesc, format: acldvppPixelFormat)
        -> i32;
    pub fn acldvppSetPicDescWidth(picDesc: *mut acldvppPicDesc, width: u32) -> i32;
    pub fn acldvppSetPicDescHeight(picDesc: *mut acldvppPicDesc, height: u32) -> i32;
    pub fn acldvppSetPicDescWidthStride(picDesc: *mut acldvppPicDesc, widthStride: u32) -> i32;
    pub fn acldvppSetPicDescHeightStride(picDesc: *mut acldvppPicDesc, heightStride: u32) -> i32;
    pub fn acldvppGetPicDescData(picDesc: *const acldvppPicDesc) -> *mut c_void;
    pub fn acldvppGetPicDescSize(picDesc: *const acldvppPicDesc) -> u32;
    pub fn acldvppGetPicDescFormat(picDesc: *const acldvppPicDesc) -> acldvppPixelFormat;
    pub fn acldvppGetPicDescWidth(picDesc: *const acldvppPicDesc) -> u32;
    pub fn acldvppGetPicDescHeight(picDesc: *const acldvppPicDesc) -> u32;
    pub fn acldvppGetPicDescWidthStride(picDesc: *const acldvppPicDesc) -> u32;
    pub fn acldvppGetPicDescHeightStride(picDesc: *const acldvppPicDesc) -> u32;
    pub fn acldvppGetPicDescRetCode(picDesc: *const acldvppPicDesc) -> u32;

    // ---- Batch picture description ----
    pub fn acldvppCreateBatchPicDesc(batchSize: u32) -> *mut acldvppBatchPicDesc;
    pub fn acldvppGetPicDesc(
        batchPicDesc: *mut acldvppBatchPicDesc,
        index: u32,
    ) -> *mut acldvppPicDesc;
    pub fn acldvppDestroyBatchPicDesc(batchPicDesc: *mut acldvppBatchPicDesc) -> i32;

    // ---- ROI config ----
    pub fn acldvppCreateRoiConfig(
        left: u32,
        right: u32,
        top: u32,
        bottom: u32,
    ) -> *mut acldvppRoiConfig;
    pub fn acldvppDestroyRoiConfig(roiConfig: *mut acldvppRoiConfig) -> i32;
    pub fn acldvppSetRoiConfig(
        config: *mut acldvppRoiConfig,
        left: u32,
        right: u32,
        top: u32,
        bottom: u32,
    ) -> i32;

    // ---- Resize config ----
    pub fn acldvppCreateResizeConfig() -> *mut acldvppResizeConfig;
    pub fn acldvppDestroyResizeConfig(resizeConfig: *mut acldvppResizeConfig) -> i32;
    pub fn acldvppSetResizeConfigInterpolation(
        resizeConfig: *mut acldvppResizeConfig,
        interpolation: u32,
    ) -> i32;
    pub fn acldvppGetResizeConfigInterpolation(resizeConfig: *const acldvppResizeConfig) -> u32;

    // ---- JPEG encode config ----
    pub fn acldvppCreateJpegeConfig() -> *mut acldvppJpegeConfig;
    pub fn acldvppDestroyJpegeConfig(jpegeConfig: *mut acldvppJpegeConfig) -> i32;
    pub fn acldvppSetJpegeConfigLevel(jpegeConfig: *mut acldvppJpegeConfig, level: u32) -> i32;
    pub fn acldvppGetJpegeConfigLevel(jpegeConfig: *const acldvppJpegeConfig) -> u32;

    // ---- VPC async operations ----
    pub fn acldvppVpcResizeAsync(
        channelDesc: *mut acldvppChannelDesc,
        inputDesc: *mut acldvppPicDesc,
        outputDesc: *mut acldvppPicDesc,
        resizeConfig: *mut acldvppResizeConfig,
        stream: aclrtStream,
    ) -> i32;

    pub fn acldvppVpcCropAsync(
        channelDesc: *mut acldvppChannelDesc,
        inputDesc: *mut acldvppPicDesc,
        outputDesc: *mut acldvppPicDesc,
        cropArea: *mut acldvppRoiConfig,
        stream: aclrtStream,
    ) -> i32;

    pub fn acldvppVpcCropAndPasteAsync(
        channelDesc: *mut acldvppChannelDesc,
        inputDesc: *mut acldvppPicDesc,
        outputDesc: *mut acldvppPicDesc,
        cropArea: *mut acldvppRoiConfig,
        pasteArea: *mut acldvppRoiConfig,
        stream: aclrtStream,
    ) -> i32;

    pub fn acldvppVpcCropResizeAsync(
        channelDesc: *mut acldvppChannelDesc,
        inputDesc: *mut acldvppPicDesc,
        outputDesc: *mut acldvppPicDesc,
        cropArea: *mut acldvppRoiConfig,
        resizeConfig: *mut acldvppResizeConfig,
        stream: aclrtStream,
    ) -> i32;

    pub fn acldvppVpcCropResizePasteAsync(
        channelDesc: *mut acldvppChannelDesc,
        inputDesc: *mut acldvppPicDesc,
        outputDesc: *mut acldvppPicDesc,
        cropArea: *mut acldvppRoiConfig,
        pasteArea: *mut acldvppRoiConfig,
        resizeConfig: *mut acldvppResizeConfig,
        stream: aclrtStream,
    ) -> i32;

    pub fn acldvppVpcConvertColorAsync(
        channelDesc: *mut acldvppChannelDesc,
        inputDesc: *mut acldvppPicDesc,
        outputDesc: *mut acldvppPicDesc,
        stream: aclrtStream,
    ) -> i32;

    // ---- JPEG decode/encode ----
    pub fn acldvppJpegDecodeAsync(
        channelDesc: *mut acldvppChannelDesc,
        data: *const c_void,
        size: u32,
        outputDesc: *mut acldvppPicDesc,
        stream: aclrtStream,
    ) -> i32;

    pub fn acldvppJpegEncodeAsync(
        channelDesc: *mut acldvppChannelDesc,
        inputDesc: *mut acldvppPicDesc,
        data: *const c_void,
        size: *mut u32,
        config: *mut acldvppJpegeConfig,
        stream: aclrtStream,
    ) -> i32;

    // ---- PNG decode ----
    pub fn acldvppPngDecodeAsync(
        channelDesc: *mut acldvppChannelDesc,
        data: *const c_void,
        size: u32,
        outputDesc: *mut acldvppPicDesc,
        stream: aclrtStream,
    ) -> i32;

    // ---- Image info queries ----
    pub fn acldvppJpegGetImageInfo(
        data: *const c_void,
        size: u32,
        width: *mut u32,
        height: *mut u32,
        components: *mut i32,
    ) -> i32;

    pub fn acldvppJpegGetImageInfoV2(
        data: *const c_void,
        size: u32,
        width: *mut u32,
        height: *mut u32,
        components: *mut i32,
        format: *mut acldvppJpegFormat,
    ) -> i32;

    pub fn acldvppPngGetImageInfo(
        data: *const c_void,
        dataSize: u32,
        width: *mut u32,
        height: *mut u32,
        components: *mut i32,
    ) -> i32;

    // ---- Size prediction ----
    pub fn acldvppJpegPredictEncSize(
        inputDesc: *const acldvppPicDesc,
        config: *const acldvppJpegeConfig,
        size: *mut u32,
    ) -> i32;

    pub fn acldvppJpegPredictDecSize(
        data: *const c_void,
        dataSize: u32,
        outputPixelFormat: acldvppPixelFormat,
        decSize: *mut u32,
    ) -> i32;

    pub fn acldvppPngPredictDecSize(
        data: *const c_void,
        dataSize: u32,
        outputPixelFormat: acldvppPixelFormat,
        decSize: *mut u32,
    ) -> i32;
}
