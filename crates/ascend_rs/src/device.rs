use std::marker::PhantomData;

use num_traits::Zero;

use crate::acl::Acl;
use crate::errors::{AclResult, ToAclResult};

/// Handler for the computing device.
#[derive(Debug)]
pub struct Device<'acl> {
    descriptor: i32,
    // Make sure that device is initialized after `Acl`.
    _acl: PhantomData<&'acl Acl>,
}

impl<'acl> Device<'acl> {
    /// Set the default device.
    pub fn new(acl: &'acl Acl) -> AclResult<Device<'acl>> {
        let dev_id = std::env::var("ASCEND_DEVICE_ID")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        Self::set_device(acl, dev_id)
    }

    /// Wrapper around `aclrtSetDevice` which specifies the device used for
    /// computation in the current thread.
    ///
    /// `aclrtSetDevice` uses a reference counting implementation. Each time
    /// the `aclrtSetDevice` interface is called, the reference count is incremented
    /// by one, and each time the `aclrtResetDevice`(see also [`Device::drop`])
    /// interface is called, the reference count is decremented by one. Only when
    /// the reference count is reduced to 0 will the resources on the device be
    /// truly released.
    pub fn set_device(_acl: &'acl Acl, descriptor: i32) -> AclResult<Self> {
        unsafe {
            ascend_sys::core::aclrtSetDevice(descriptor).to_result()?;
            Ok(Device::from_raw(descriptor))
        }
    }

    /// Enables data interaction between the current device and the specified device.
    ///
    /// You can call the [`Device::can_access_peer`] interface in advance to check
    /// whether data exchange is possible between the current device and a specified device.
    /// If you want to disable data exchange between devices after enabling it, you can
    /// call the  [`Device::disable_peer_access`] interface.
    pub fn enable_peer_access(_acl: &'acl Acl, peer: &Device<'acl>) -> AclResult<()> {
        unsafe {
            // `flags` parameter is reserved and must currently be set to 0.
            ascend_sys::core::aclrtDeviceEnablePeerAccess(peer.descriptor, 0).to_result()?;
            Ok(())
        }
    }

    /// Disables data interaction between the current device and the specified device.
    ///
    /// See also [`Device::enable_peer_access`].
    pub fn disable_peer_access(_acl: &'acl Acl, peer: &Device<'acl>) -> AclResult<()> {
        unsafe {
            ascend_sys::core::aclrtDeviceDisablePeerAccess(peer.descriptor).to_result()?;
            Ok(())
        }
    }

    /// Creates a device from a raw object.
    ///
    /// # Safety
    ///
    /// A raw device must be valid.
    pub unsafe fn from_raw(descriptor: i32) -> Self {
        Self {
            descriptor,
            _acl: PhantomData::default(),
        }
    }

    /// Converts a device into a raw object.
    pub fn to_raw(&self) -> i32 {
        self.descriptor
    }

    /// Check whether data exchange is supported between devices.
    pub fn can_access_peer(&self, peer: &Device<'acl>) -> AclResult<bool> {
        unsafe {
            let mut res = 0;
            ascend_sys::core::aclrtDeviceCanAccessPeer(
                std::ptr::addr_of_mut!(res),
                self.descriptor,
                peer.descriptor,
            )
            .to_result()?;
            Ok(!res.is_zero())
        }
    }

    /// Get the descriptor (device ID) of this device.
    pub fn descriptor(&self) -> i32 {
        self.descriptor
    }
}

/// Query the number of available devices.
///
/// This function can be called before [`Device::new`] to discover available hardware.
pub fn device_count() -> AclResult<u32> {
    unsafe {
        let mut count: u32 = 0;
        ascend_sys::core::aclrtGetDeviceCount(&mut count).to_result()?;
        Ok(count)
    }
}

/// Information about device memory usage.
#[derive(Debug, Clone, Copy)]
pub struct MemInfo {
    /// Free memory in bytes.
    pub free: usize,
    /// Total memory in bytes.
    pub total: usize,
}

impl MemInfo {
    /// Returns the used memory in bytes.
    pub fn used(&self) -> usize {
        self.total - self.free
    }
}

/// Query free and total device memory for the current device.
///
/// Returns a [`MemInfo`] struct with `free` and `total` fields in bytes.
pub fn mem_info() -> AclResult<MemInfo> {
    unsafe {
        let mut free: usize = 0;
        let mut total: usize = 0;
        ascend_sys::core::aclrtGetMemInfo(
            ascend_sys::core::aclrtMemAttr_ACL_DDR_MEM,
            &mut free,
            &mut total,
        )
        .to_result()?;
        Ok(MemInfo { free, total })
    }
}

/// Query HBM (High Bandwidth Memory) info for the current device.
///
/// Returns a [`MemInfo`] struct with `free` and `total` fields in bytes.
/// Not all devices have HBM; returns an error if unavailable.
pub fn hbm_info() -> AclResult<MemInfo> {
    unsafe {
        let mut free: usize = 0;
        let mut total: usize = 0;
        ascend_sys::core::aclrtGetMemInfo(
            ascend_sys::core::aclrtMemAttr_ACL_HBM_MEM,
            &mut free,
            &mut total,
        )
        .to_result()?;
        Ok(MemInfo { free, total })
    }
}

/// RAII wrapper around `aclrtResetDevice`. See [`Device::set_device`] for
/// more details.
impl Drop for Device<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = ascend_sys::core::aclrtResetDevice(self.descriptor);
        }
    }
}
