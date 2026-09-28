use std::fmt;

/// Ascend NPU target SoC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AscendTarget {
    Ascend310,
    Ascend310B,
    Ascend310P1,
    Ascend310P3,
    Ascend910B1,
    Ascend910B2,
    Ascend910B3,
    Ascend910B4,
}

/// Output format for the compiled kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Relocatable object file (`-c -o out.o`)
    Object,
    /// Shared library (`-fPIC --shared -o out.so`)
    SharedLib,
}

/// Which flag convention to use when invoking bisheng.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlagStyle {
    /// `--cce-aicore-arch=<arch>` — used for 310P, 310B, 310, 910Pro
    CceAicore,
    /// `--npu-arch=<arch> -xasc` — used for 910B (TileLang-compatible)
    NpuArch,
}

impl AscendTarget {
    /// Parse a SoC version string (e.g. `"Ascend910B3"`, `"Ascend310P1"`).
    pub fn from_soc_version(s: &str) -> Option<AscendTarget> {
        match s {
            "Ascend310" => Some(AscendTarget::Ascend310),
            "Ascend310B" | "Ascend310B1" | "Ascend310B2" | "Ascend310B3" | "Ascend310B4" => {
                Some(AscendTarget::Ascend310B)
            }
            "Ascend310P1" | "Ascend310P" => Some(AscendTarget::Ascend310P1),
            "Ascend310P3" => Some(AscendTarget::Ascend310P3),
            "Ascend910B1" => Some(AscendTarget::Ascend910B1),
            "Ascend910B2" => Some(AscendTarget::Ascend910B2),
            "Ascend910B3" => Some(AscendTarget::Ascend910B3),
            "Ascend910B4" => Some(AscendTarget::Ascend910B4),
            _ => None,
        }
    }

    /// The `--cce-aicore-arch` value for this target.
    /// Maps from `RunMode::cce_aicore_arch()` in tile_kernel_builder_config.
    pub fn cce_aicore_arch(&self) -> &'static str {
        match self {
            AscendTarget::Ascend310 => "dav-c100",
            AscendTarget::Ascend310B => "dav-m200i",
            AscendTarget::Ascend310P1 | AscendTarget::Ascend310P3 => "dav-m200",
            AscendTarget::Ascend910B1
            | AscendTarget::Ascend910B2
            | AscendTarget::Ascend910B3
            | AscendTarget::Ascend910B4 => "dav-c220",
        }
    }

    /// The `--npu-arch` value, if supported. Only 910B targets use this style.
    pub fn npu_arch(&self) -> Option<&'static str> {
        match self {
            AscendTarget::Ascend910B1
            | AscendTarget::Ascend910B2
            | AscendTarget::Ascend910B3
            | AscendTarget::Ascend910B4 => Some("dav-2201"),
            _ => None,
        }
    }

    /// The default flag style for this target.
    pub fn default_flag_style(&self) -> FlagStyle {
        match self.npu_arch() {
            Some(_) => FlagStyle::NpuArch,
            None => FlagStyle::CceAicore,
        }
    }

    /// Whether this is a 910B-series target (supports cube engine dual compilation).
    pub fn is_910b(&self) -> bool {
        matches!(
            self,
            AscendTarget::Ascend910B1
                | AscendTarget::Ascend910B2
                | AscendTarget::Ascend910B3
                | AscendTarget::Ascend910B4
        )
    }

    /// Whether this target requires explicit `pipe_barrier()` calls
    /// (310P has no auto-sync; 910B can rely on compiler auto-sync).
    pub fn needs_explicit_barriers(&self) -> bool {
        match self {
            AscendTarget::Ascend310
            | AscendTarget::Ascend310B
            | AscendTarget::Ascend310P1
            | AscendTarget::Ascend310P3 => true,
            _ => false,
        }
    }

    /// Whether this target uses the C++ codegen path
    /// (310-family requires bisheng C++ path, not bishengir-compile).
    pub fn needs_cpp_codegen(&self) -> bool {
        match self {
            AscendTarget::Ascend310
            | AscendTarget::Ascend310B
            | AscendTarget::Ascend310P1
            | AscendTarget::Ascend310P3 => true,
            _ => false,
        }
    }

    /// SoC version string (canonical form).
    pub fn soc_version(&self) -> &'static str {
        match self {
            AscendTarget::Ascend310 => "Ascend310",
            AscendTarget::Ascend310B => "Ascend310B",
            AscendTarget::Ascend310P1 => "Ascend310P1",
            AscendTarget::Ascend310P3 => "Ascend310P3",
            AscendTarget::Ascend910B1 => "Ascend910B1",
            AscendTarget::Ascend910B2 => "Ascend910B2",
            AscendTarget::Ascend910B3 => "Ascend910B3",
            AscendTarget::Ascend910B4 => "Ascend910B4",
        }
    }

    /// SoC version string in lowercase, matching CANN's expected format for
    /// `AscendCheckSoCVersion()` and `.ascend.kernel.<soc>.<name>` section names.
    pub fn soc_version_lower(&self) -> &'static str {
        match self {
            AscendTarget::Ascend310 => "ascend310",
            AscendTarget::Ascend310B => "ascend310b",
            AscendTarget::Ascend310P1 => "ascend310p1",
            AscendTarget::Ascend310P3 => "ascend310p3",
            AscendTarget::Ascend910B1 => "ascend910b1",
            AscendTarget::Ascend910B2 => "ascend910b2",
            AscendTarget::Ascend910B3 => "ascend910b3",
            AscendTarget::Ascend910B4 => "ascend910b4",
        }
    }

    /// Unified Buffer size in bytes.
    pub fn ub_size(&self) -> usize {
        match self {
            AscendTarget::Ascend310P1 | AscendTarget::Ascend310P3 => 256 * 1024,
            _ => 192 * 1024,
        }
    }

    /// L1 buffer size in bytes.
    pub fn l1_size(&self) -> usize {
        512 * 1024
    }

    /// L0A buffer size in bytes.
    pub fn l0a_size(&self) -> usize {
        64 * 1024
    }

    /// L0B buffer size in bytes.
    pub fn l0b_size(&self) -> usize {
        64 * 1024
    }

    /// L0C buffer size in bytes.
    pub fn l0c_size(&self) -> usize {
        128 * 1024
    }
}

impl fmt::Display for AscendTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.soc_version())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_soc_versions() {
        assert_eq!(
            AscendTarget::from_soc_version("Ascend910B3"),
            Some(AscendTarget::Ascend910B3)
        );
        assert_eq!(
            AscendTarget::from_soc_version("Ascend310P1"),
            Some(AscendTarget::Ascend310P1)
        );
        assert_eq!(
            AscendTarget::from_soc_version("Ascend310P"),
            Some(AscendTarget::Ascend310P1)
        );
        assert_eq!(AscendTarget::from_soc_version("Unknown"), None);
    }

    #[test]
    fn flag_styles() {
        assert_eq!(
            AscendTarget::Ascend910B3.default_flag_style(),
            FlagStyle::NpuArch
        );
        assert_eq!(
            AscendTarget::Ascend310P1.default_flag_style(),
            FlagStyle::CceAicore
        );
    }

    #[test]
    fn hardware_limits() {
        assert_eq!(AscendTarget::Ascend310P1.ub_size(), 256 * 1024);
        assert_eq!(AscendTarget::Ascend910B3.ub_size(), 192 * 1024);
    }

    #[test]
    fn barrier_requirements() {
        assert!(AscendTarget::Ascend310P1.needs_explicit_barriers());
        assert!(!AscendTarget::Ascend910B3.needs_explicit_barriers());
    }
}
