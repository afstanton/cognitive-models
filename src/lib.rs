pub mod model;
pub mod models;

use candle_core::{Device, Result};

pub use model::{CognitiveLM, ModelConfig};

pub fn preferred_device() -> Result<Device> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        Device::new_metal(0).or(Ok(Device::Cpu))
    }

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        Device::new_cuda(0).or(Ok(Device::Cpu))
    }

    #[cfg(not(any(
        all(target_os = "macos", feature = "metal"),
        all(not(target_os = "macos"), feature = "cuda")
    )))]
    {
        Ok(Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::preferred_device;

    #[test]
    fn preferred_device_resolves() {
        let _ = preferred_device().expect("preferred device should resolve");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn preferred_device_uses_metal_on_macos_when_available() {
        let device = preferred_device().expect("preferred device should resolve on macOS");
        assert!(
            device.is_metal(),
            "expected Metal device on macOS builds with metal feature"
        );
    }
}
