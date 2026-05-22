use anyhow::{Result, anyhow};
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
}

/// Runtime-resolved backend. Variants are feature-gated to match what was
/// actually compiled in, so the dispatch `match` stays exhaustive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedDevice {
    Cpu,
    #[cfg(feature = "gpu")]
    Wgpu,
}

/// Map a `DeviceChoice` to a `ResolvedDevice`, erroring with a rebuild hint
/// when the requested backend was not compiled in.
///
/// `Auto` prefers WGPU when the `gpu` feature is available, otherwise CPU.
/// Real adapter probing (with Windows fallback) lands in a later phase E commit.
pub fn resolve(choice: DeviceChoice) -> Result<ResolvedDevice> {
    match choice {
        DeviceChoice::Auto => {
            #[cfg(feature = "gpu")]
            {
                Ok(ResolvedDevice::Wgpu)
            }
            #[cfg(not(feature = "gpu"))]
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
                Err(anyhow!(
                    "WGPU backend not compiled in. Rebuild with the default features or `--features gpu`."
                ))
            }
        }
        DeviceChoice::Cuda => Err(anyhow!(
            "CUDA backend not enabled. Rebuild with `--features cuda`."
        )),
    }
}
