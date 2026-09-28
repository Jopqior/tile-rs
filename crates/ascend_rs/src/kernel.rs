use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::fs;
use std::io::Read;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::OnceLock;

use crate::errors::{AclInnerError, AclResult, ToAclResult};
use crate::stream::AclStream;

// ---------------------------------------------------------------------------
// Runtime function loading via dlopen/dlsym (libloading).
//
// We cannot directly link libruntime.so (-lruntime) because its constructors
// run at process startup and interfere with aclInit, causing it to return
// ACL_ERROR_INTERNAL_ERROR (500000). Instead, we load libruntime.so lazily
// via dlopen after aclInit has already succeeded.
// ---------------------------------------------------------------------------

type RtDevBinaryRegisterFn = unsafe extern "C" fn(
    bin: *const ascend_sys::core::tagRtDevBinary,
    handle: *mut *mut c_void,
) -> i32;

type RtFunctionRegisterFn = unsafe extern "C" fn(
    handle: *mut c_void,
    func_name: *const c_char,
    dev_func: *const c_char,
    stub_func: *const c_char,
    flag: u32,
) -> i32;

type RtKernelLaunchFn = unsafe extern "C" fn(
    stub_func: *const c_void,
    block_dim: u32,
    args: *mut c_void,
    args_size: u32,
    sm_desc: *mut c_void,
    stream: *mut c_void,
) -> i32;

/// rtGetC2cCtrlAddr(void **addr, uint32_t *prefCnt) -> int32_t
/// Returns the FFTS (Fast Task Scheduling) sync address for cube engine initialization.
type RtGetC2cCtrlAddrFn = unsafe extern "C" fn(addr: *mut *mut c_void, pref_cnt: *mut u32) -> i32;

struct RuntimeFns {
    _lib: libloading::Library,
    rt_dev_binary_register: RtDevBinaryRegisterFn,
    rt_function_register: RtFunctionRegisterFn,
    rt_kernel_launch: RtKernelLaunchFn,
    rt_get_c2c_ctrl_addr: Option<RtGetC2cCtrlAddrFn>,
}

unsafe impl Send for RuntimeFns {}
unsafe impl Sync for RuntimeFns {}

static RUNTIME_FNS: OnceLock<RuntimeFns> = OnceLock::new();

fn runtime() -> &'static RuntimeFns {
    RUNTIME_FNS.get_or_init(|| unsafe {
        let lib =
            libloading::Library::new("libruntime.so").expect("Failed to dlopen libruntime.so");
        let rt_dev_binary_register: RtDevBinaryRegisterFn = *lib
            .get(b"rtDevBinaryRegister\0")
            .expect("rtDevBinaryRegister not found");
        let rt_function_register: RtFunctionRegisterFn = *lib
            .get(b"rtFunctionRegister\0")
            .expect("rtFunctionRegister not found");
        let rt_kernel_launch: RtKernelLaunchFn = *lib
            .get(b"rtKernelLaunch\0")
            .expect("rtKernelLaunch not found");
        let rt_get_c2c_ctrl_addr: Option<RtGetC2cCtrlAddrFn> =
            lib.get(b"rtGetC2cCtrlAddr\0").ok().map(|f| *f);
        RuntimeFns {
            _lib: lib,
            rt_dev_binary_register,
            rt_function_register,
            rt_kernel_launch,
            rt_get_c2c_ctrl_addr,
        }
    })
}

// ---------------------------------------------------------------------------

/// Selects the kernel binary type for runtime registration.
///
/// Different kernel types use different AICore execution pipelines:
/// - `Vector`: Vector engine (default for element-wise ops)
/// - `Cube`: Cube engine (for matmul/Mmad operations using L0A/L0B/L0C)
/// - `AiCore`: General AICore (both vector and cube)
#[derive(Debug, Clone, Copy, Default)]
pub enum KernelMagic {
    #[default]
    Vector,
    Cube,
    AiCore,
}

pub struct KernelLoader {
    handle: *mut c_void,
    /// Maps kernel name → registered CString. The CString's pointer is used as
    /// the `stubFunc` for rtKernelLaunch and must remain stable across calls.
    pool: RefCell<HashMap<String, CString>>,
    /// When loaded from a .so, keeps the library open to prevent dlclose.
    _lib: Option<libloading::Library>,
}

impl KernelLoader {
    pub fn new() -> AclResult<KernelLoader> {
        let out_dir: PathBuf = std::env::var("OUT_DIR")
            .expect("`OUT_DIR` is not specified for kernel default path")
            .into();
        let default_path = out_dir.join("kernel.o");
        Self::from_bin_path(default_path)
    }

    pub fn from_bin_path(path: impl AsRef<Path>) -> AclResult<KernelLoader> {
        // Auto-detect magic: if binary contains _mix_aic symbols (cube engine),
        // use AiCore (general) magic. Otherwise use Vector magic.
        let bytes = fs::read(path.as_ref())
            .expect("`KernelLoader::from_bin_path`: cannot read kernel file");
        let magic = if bytes.windows(8).any(|w| w == b"_mix_aic") {
            KernelMagic::AiCore
        } else {
            KernelMagic::Vector
        };
        Self::from_bin_path_with_magic(path, magic)
    }

    pub fn from_bin_path_with_magic(
        path: impl AsRef<Path>,
        magic: KernelMagic,
    ) -> AclResult<KernelLoader> {
        let kernel_path = path.as_ref();
        let mut file = fs::File::open(&kernel_path)
            .expect("`KernelLoader::from_bin_path`: no kernel file found");
        let metadata = fs::metadata(&kernel_path)
            .expect("`KernelLoader::from_bin_path`: unable to open file metadata");
        let mut buffer = Vec::with_capacity(metadata.len() as usize);
        buffer.resize(metadata.len() as usize, 0);
        let _ = file
            .read(buffer.as_mut_slice())
            .expect("`KernelLoader::from_bin_path`: buffer overflow");

        let mut handle: *mut c_void = std::ptr::null_mut();

        // Magic values from CANN SDK kernel.h:
        //   RT_DEV_BINARY_MAGIC_ELF       = 0x43554245 (AICore general)
        //   RT_DEV_BINARY_MAGIC_ELF_AIVEC = 0x41415246 (AI Vector)
        //   RT_DEV_BINARY_MAGIC_ELF_AICUBE = 0x41494343 (AI Cube)
        let magic_value = match magic {
            KernelMagic::Vector => 0x41415246u32,
            KernelMagic::Cube => 0x41494343u32,
            KernelMagic::AiCore => 0x43554245u32,
        };

        let bin = ascend_sys::core::tagRtDevBinary {
            data: buffer.as_ptr() as *const _,
            length: buffer.len() as u64,
            magic: magic_value,
            version: 0,
        };

        let rt = runtime();
        unsafe { (rt.rt_dev_binary_register)(&bin, &mut handle).to_result()? };
        Ok(KernelLoader {
            handle,
            pool: Default::default(),
            _lib: None,
        })
    }

    /// Load a cube kernel from a shared library (.so) produced by the
    /// AIC+AIV dual compilation pipeline.
    ///
    /// The .so contains a constructor (`__register_kernels`) that calls
    /// `RegisterAscendBinary` to register the device binary. After dlopen,
    /// the kernel handle is available via the `g_kernel_handle` global.
    pub fn from_shared_lib(path: impl AsRef<Path>) -> AclResult<KernelLoader> {
        let lib = unsafe { libloading::Library::new(path.as_ref().as_os_str()) }.map_err(|e| {
            eprintln!("KernelLoader::from_shared_lib: dlopen failed: {}", e);
            AclInnerError::AclInnerErrorInternalError
        })?;

        // The constructor has already run. Read g_kernel_handle.
        let handle: *mut c_void = unsafe {
            let handle_ptr: libloading::Symbol<*mut *mut c_void> =
                lib.get(b"g_kernel_handle\0").map_err(|e| {
                    eprintln!(
                        "KernelLoader::from_shared_lib: g_kernel_handle not found: {}",
                        e
                    );
                    AclInnerError::AclInnerErrorInternalError
                })?;
            **handle_ptr
        };

        if handle.is_null() {
            eprintln!(
                "KernelLoader::from_shared_lib: g_kernel_handle is null (RegisterAscendBinary failed?)"
            );
            return Err(AclInnerError::AclInnerErrorInternalError.into());
        }

        Ok(KernelLoader {
            handle,
            pool: Default::default(),
            _lib: Some(lib),
        })
    }

    pub fn get_kernel<'a>(&'a self, name: &str) -> AclResult<Kernel<'a>> {
        let mut pool = self.pool.borrow_mut();
        if !pool.contains_key(name) {
            let c_name = CString::from_str(name).unwrap();
            // Register the kernel with the runtime. The CString's pointer is
            // used as the stubFunc identifier — it must remain stable.
            let rt = runtime();
            let ret = unsafe {
                (rt.rt_function_register)(
                    self.handle,
                    c_name.as_ptr() as *const _,
                    c_name.as_ptr() as *const _,
                    c_name.as_ptr() as *const _,
                    0,
                )
            };
            let err = AclInnerError::from(ret);
            match err {
                AclInnerError::AclSuccess | AclInnerError::AclInnerErrorRtKernelDuplicate => {}
                _ => ret.to_result()?,
            }
            pool.insert(name.to_string(), c_name);
        }
        // Return a Kernel that borrows the stable CString from the pool.
        // The pointer passed to rtKernelLaunch must match rtFunctionRegister.
        let c_name = pool.get(name).unwrap();
        Ok(Kernel::from_ptr(c_name.as_ptr()))
    }
}

pub struct Kernel<'kl> {
    /// Raw pointer to the CString stored in KernelLoader's pool.
    /// Must be the same pointer passed to rtFunctionRegister.
    stub_ptr: *const c_char,
    _module: PhantomData<&'kl KernelLoader>,
}

impl<'kl> Kernel<'kl> {
    fn from_ptr(ptr: *const c_char) -> Self {
        Self {
            stub_ptr: ptr,
            _module: PhantomData,
        }
    }

    pub unsafe fn launch(
        &self,
        dim: u32,
        stream: &AclStream,
        args: &mut [*mut c_void],
    ) -> AclResult<()> {
        let rt = runtime();
        unsafe {
            // Launch kernel on `dim` AICore blocks in a single call.
            // rtKernelLaunch with block_dim=N dispatches N blocks internally.
            (rt.rt_kernel_launch)(
                self.stub_ptr as *const _,
                dim,
                args.as_mut_ptr() as *mut _,
                (args.len() * size_of::<*mut c_void>()) as u32,
                std::ptr::null_mut(),
                stream.to_raw(),
            )
            .to_result()?;
        }
        Ok(())
    }

    /// Launch a cube kernel with FFTS wrapper args.
    ///
    /// Cube kernels on 910B require `ffts_addr` (first arg) and `overflow_status`
    /// (last arg) for proper cube engine initialization. This method:
    /// 1. Gets the FFTS sync address via `rtGetC2cCtrlAddr`
    /// 2. Allocates an overflow_status buffer (8 bytes)
    /// 3. Prepends ffts_addr and appends overflow_status to the args
    /// 4. Launches the kernel
    pub unsafe fn launch_cube(
        &self,
        dim: u32,
        stream: &AclStream,
        args: &mut [*mut c_void],
    ) -> AclResult<()> {
        let ffts_addr = get_ffts_addr();
        // Allocate overflow status buffer (8 bytes on device)
        let mut overflow: *mut c_void = std::ptr::null_mut();
        unsafe {
            let _ = ascend_sys::core::aclrtMalloc(
                &mut overflow as *mut *mut c_void,
                8,
                0u32, // ACL_MEM_MALLOC_HUGE_FIRST
            );
        }

        // Build wrapped args: [ffts_addr, ...original_args..., overflow_status]
        let mut wrapped_args: Vec<*mut c_void> = Vec::with_capacity(args.len() + 2);
        wrapped_args.push(ffts_addr);
        wrapped_args.extend_from_slice(args);
        wrapped_args.push(overflow);

        let rt = runtime();
        let result = unsafe {
            (rt.rt_kernel_launch)(
                self.stub_ptr as *const _,
                dim,
                wrapped_args.as_mut_ptr() as *mut _,
                (wrapped_args.len() * size_of::<*mut c_void>()) as u32,
                std::ptr::null_mut(),
                stream.to_raw(),
            )
            .to_result()
        };

        // Free overflow buffer after launch (async — kernel has already read it)
        if !overflow.is_null() {
            unsafe {
                ascend_sys::core::aclrtFree(overflow);
            }
        }

        result
    }
}

/// Get the FFTS (Fast Task Scheduling) sync address for cube engine initialization.
/// Returns null if not available (non-910B targets).
pub fn get_ffts_addr() -> *mut c_void {
    let rt = runtime();
    if let Some(get_addr) = rt.rt_get_c2c_ctrl_addr {
        let mut addr: *mut c_void = std::ptr::null_mut();
        let mut pref_cnt: u32 = 0;
        let ret = unsafe { get_addr(&mut addr, &mut pref_cnt) };
        if ret == 0 {
            return addr;
        }
    }
    std::ptr::null_mut()
}
