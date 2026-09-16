use std::ffi::{CStr, CString, c_char, c_int};

use crate::compiler::CompileConfig;
use crate::target::{AscendTarget, FlagStyle, OutputFormat};

/// Opaque result handle returned to C/Python callers.
pub struct CompileResult {
    data: Vec<u8>,
    error: Option<CString>,
}

/// Create a new compile config for the given SoC version string.
/// Returns null if the SoC is not recognized.
///
/// # Safety
/// `soc` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_config_new(soc: *const c_char) -> *mut CompileConfig {
    let soc_str = match unsafe { CStr::from_ptr(soc) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let target = match AscendTarget::from_soc_version(soc_str) {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(CompileConfig::new(target)))
}

/// Set the output format: 0 = Object, 1 = SharedLib.
///
/// # Safety
/// `cfg` must be a valid pointer from `ascend_compile_config_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_config_set_output_format(
    cfg: *mut CompileConfig,
    fmt: c_int,
) {
    let cfg = unsafe { &mut *cfg };
    cfg.output_format = match fmt {
        1 => OutputFormat::SharedLib,
        _ => OutputFormat::Object,
    };
}

/// Set the flag style: 0 = CceAicore, 1 = NpuArch.
///
/// # Safety
/// `cfg` must be a valid pointer from `ascend_compile_config_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_config_set_flag_style(
    cfg: *mut CompileConfig,
    style: c_int,
) {
    let cfg = unsafe { &mut *cfg };
    cfg.flag_style = match style {
        1 => FlagStyle::NpuArch,
        _ => FlagStyle::CceAicore,
    };
}

/// Set the optimization level (0-3).
///
/// # Safety
/// `cfg` must be a valid pointer from `ascend_compile_config_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_config_set_opt_level(
    cfg: *mut CompileConfig,
    level: c_int,
) {
    let cfg = unsafe { &mut *cfg };
    cfg.opt_level = level.clamp(0, 3) as u8;
}

/// Disable or enable validation (1 = validate, 0 = skip).
///
/// # Safety
/// `cfg` must be a valid pointer from `ascend_compile_config_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_config_set_validate(
    cfg: *mut CompileConfig,
    validate: c_int,
) {
    let cfg = unsafe { &mut *cfg };
    cfg.validate = validate != 0;
}

/// Add an include path.
///
/// # Safety
/// `cfg` must be a valid pointer. `path` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_config_add_include(
    cfg: *mut CompileConfig,
    path: *const c_char,
) {
    let cfg = unsafe { &mut *cfg };
    if let Ok(s) = unsafe { CStr::from_ptr(path) }.to_str() {
        cfg.extra_includes.push(s.to_string());
    }
}

/// Add a preprocessor define.
///
/// # Safety
/// `cfg` must be a valid pointer. `def` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_config_add_define(
    cfg: *mut CompileConfig,
    def: *const c_char,
) {
    let cfg = unsafe { &mut *cfg };
    if let Ok(s) = unsafe { CStr::from_ptr(def) }.to_str() {
        cfg.extra_defines.push(s.to_string());
    }
}

/// Add a link library (SharedLib output only).
///
/// # Safety
/// `cfg` must be a valid pointer. `lib` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_config_add_lib(
    cfg: *mut CompileConfig,
    lib: *const c_char,
) {
    let cfg = unsafe { &mut *cfg };
    if let Ok(s) = unsafe { CStr::from_ptr(lib) }.to_str() {
        cfg.extra_libs.push(s.to_string());
    }
}

/// Free a compile config.
///
/// # Safety
/// `cfg` must be a valid pointer from `ascend_compile_config_new`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_config_free(cfg: *mut CompileConfig) {
    if !cfg.is_null() {
        drop(unsafe { Box::from_raw(cfg) });
    }
}

/// Compile a C++ kernel from source.
///
/// Returns a `CompileResult` handle. Check `ascend_compile_result_error()` for errors.
///
/// # Safety
/// `src` must point to `len` bytes of valid UTF-8. `cfg` must be a valid config pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_kernel(
    src: *const u8,
    len: usize,
    cfg: *const CompileConfig,
) -> *mut CompileResult {
    let source = match std::str::from_utf8(unsafe { std::slice::from_raw_parts(src, len) }) {
        Ok(s) => s,
        Err(e) => {
            return Box::into_raw(Box::new(CompileResult {
                data: Vec::new(),
                error: Some(
                    CString::new(format!("invalid UTF-8 source: {}", e)).unwrap_or_default(),
                ),
            }));
        }
    };

    let config = unsafe { &*cfg };
    match crate::compile_kernel(source, config) {
        Ok(data) => Box::into_raw(Box::new(CompileResult { data, error: None })),
        Err(e) => Box::into_raw(Box::new(CompileResult {
            data: Vec::new(),
            error: Some(CString::new(e.to_string()).unwrap_or_default()),
        })),
    }
}

/// Get the compiled binary data pointer. Returns null if compilation failed.
///
/// # Safety
/// `r` must be a valid `CompileResult` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_result_data(r: *const CompileResult) -> *const u8 {
    let r = unsafe { &*r };
    if r.error.is_some() || r.data.is_empty() {
        std::ptr::null()
    } else {
        r.data.as_ptr()
    }
}

/// Get the compiled binary length. Returns 0 if compilation failed.
///
/// # Safety
/// `r` must be a valid `CompileResult` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_result_len(r: *const CompileResult) -> usize {
    let r = unsafe { &*r };
    if r.error.is_some() { 0 } else { r.data.len() }
}

/// Get the error message, or null if compilation succeeded.
///
/// # Safety
/// `r` must be a valid `CompileResult` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_result_error(r: *const CompileResult) -> *const c_char {
    let r = unsafe { &*r };
    match &r.error {
        Some(e) => e.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Free a compile result.
///
/// # Safety
/// `r` must be a valid `CompileResult` pointer, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ascend_compile_result_free(r: *mut CompileResult) {
    if !r.is_null() {
        drop(unsafe { Box::from_raw(r) });
    }
}
