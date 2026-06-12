use anyhow::Result;
use clap::ValueEnum;

/// User-facing device selection. Always exposes the full set so the error
/// message for a feature-gated backend can point at the specific rebuild
/// flag instead of a generic "invalid value".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "lowercase")]
pub enum DeviceChoice {
    #[default]
    Auto,
    Cpu,
    Wgpu,
    Cuda,
    Metal,
    Vulkan,
}

/// Runtime-resolved backend. Variants are feature-gated to match what was
/// actually compiled in, so the dispatch `match` stays exhaustive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedDevice {
    Cpu,
    #[cfg(feature = "gpu")]
    Wgpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(target_os = "macos")]
    Metal,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    Vulkan,
}

/// Map a `DeviceChoice` to a `ResolvedDevice`, erroring with a rebuild hint
/// when the requested backend was not compiled in.
///
/// `Auto` prefers CUDA when compiled in, then WGPU, then CPU. Real adapter
/// probing (with Windows fallback) lands in a later phase E commit.
pub fn resolve(choice: DeviceChoice) -> Result<ResolvedDevice> {
    match choice {
        DeviceChoice::Auto => {
            #[cfg(feature = "cuda")]
            {
                Ok(ResolvedDevice::Cuda)
            }
            #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
            {
                Ok(ResolvedDevice::Metal)
            }
            #[cfg(all(
                not(feature = "cuda"),
                not(target_os = "macos"),
                any(target_os = "linux", target_os = "windows")
            ))]
            {
                Ok(ResolvedDevice::Vulkan)
            }
            #[cfg(all(
                not(feature = "cuda"),
                not(target_os = "macos"),
                not(any(target_os = "linux", target_os = "windows")),
                feature = "gpu"
            ))]
            {
                Ok(ResolvedDevice::Wgpu)
            }
            #[cfg(all(
                not(feature = "cuda"),
                not(target_os = "macos"),
                not(any(target_os = "linux", target_os = "windows")),
                not(feature = "gpu")
            ))]
            {
                Ok(ResolvedDevice::Cpu)
            }
        }
        DeviceChoice::Cpu => Ok(ResolvedDevice::Cpu),
        DeviceChoice::Wgpu => {
            #[cfg(feature = "gpu")]
            {
                Ok(ResolvedDevice::Wgpu)
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err(anyhow::anyhow!(
                    "WGPU backend not compiled in. Rebuild with the default features or `--features gpu`."
                ))
            }
        }
        DeviceChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(ResolvedDevice::Cuda)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow::anyhow!(
                    "CUDA backend not enabled. Rebuild with `--features cuda`."
                ))
            }
        }
        DeviceChoice::Metal => {
            #[cfg(target_os = "macos")]
            {
                Ok(ResolvedDevice::Metal)
            }
            #[cfg(not(target_os = "macos"))]
            {
                Err(anyhow::anyhow!(
                    "Metal backend is only available on macOS (Apple)."
                ))
            }
        }
        DeviceChoice::Vulkan => {
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                Ok(ResolvedDevice::Vulkan)
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                Err(anyhow::anyhow!(
                    "Vulkan backend ships on Linux/Windows; on macOS use --device metal."
                ))
            }
        }
    }
}
