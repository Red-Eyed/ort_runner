//! What environment would a run actually happen in?
//!
//! Gathering is separated from rendering: `gather()` returns a plain `Serialize` value, and the
//! human and JSON outputs are two renderings of it rather than two print paths that can drift.
//!
//! Everything ONNX-Runtime-related is read from the *loaded* runtime rather than inferred from
//! the build target. The C++ decided provider availability at compile time with
//! `#ifdef __ANDROID__`, which encoded a belief that silently went stale: a comment asserted
//! XNNPACK was available in the Linux prebuilts, but those ship no XNNPACK kernels. Asking the
//! runtime cannot drift, and it stays correct when the bundled library is swapped.

pub mod platform;
pub mod render;

use std::path::Path;

use anyhow::Result;
use ort::ep::{WebGPU, CPU, NNAPI, XNNPACK};
use ort::execution_providers::ExecutionProvider;
use serde::Serialize;

use crate::cli::Provider;
use platform::{DeviceIdentity, DeviceProbe};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ProviderStatus {
    pub provider: Provider,
    pub available: bool,
}

/// A compute device as ONNX Runtime itself sees it.
///
/// Preferred over probing the GPU through EGL/Vulkan: what matters for a benchmark is what the
/// runtime will actually dispatch to, not what silicon is physically present.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeDevice {
    pub kind: String,
    pub id: u32,
    pub execution_provider: Option<String>,
    pub vendor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuInfo {
    pub arch: String,
    pub logical_cores: usize,
    pub brand: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostInfo {
    pub os: Option<String>,
    pub os_version: Option<String>,
    pub kernel: Option<String>,
    pub hostname: Option<String>,
    pub total_memory_bytes: u64,
    pub cpu: CpuInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub ort_runner_version: String,
    pub onnxruntime_build: String,
    pub dylib_path: String,
    pub platform: String,
    pub device: DeviceIdentity,
    pub host: HostInfo,
    pub providers: Vec<ProviderStatus>,
    pub runtime_devices: Vec<RuntimeDevice>,
}

/// Collects everything `--info` reports.
///
/// The probe is injected rather than constructed here so this stays testable with a fake and so
/// the platform choice lives at the edge (`platform::probe()`), not buried in the middle.
///
/// # Errors
/// If the loaded runtime rejects an execution-provider availability query.
pub fn gather(probe: &dyn DeviceProbe, dylib_path: &Path) -> Result<SystemInfo> {
    Ok(SystemInfo {
        ort_runner_version: env!("CARGO_PKG_VERSION").to_string(),
        onnxruntime_build: ort::info().to_string(),
        dylib_path: dylib_path.display().to_string(),
        platform: probe.platform().to_string(),
        device: probe.identity(),
        host: host_info(),
        providers: provider_statuses()?,
        runtime_devices: runtime_devices(),
    })
}

fn host_info() -> HostInfo {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    system.refresh_cpu_all();

    HostInfo {
        os: sysinfo::System::name(),
        os_version: sysinfo::System::os_version(),
        kernel: sysinfo::System::kernel_version(),
        hostname: sysinfo::System::host_name(),
        total_memory_bytes: system.total_memory(),
        cpu: CpuInfo {
            arch: sysinfo::System::cpu_arch(),
            logical_cores: system.cpus().len(),
            brand: system
                .cpus()
                .first()
                .map(|cpu| cpu.brand().trim().to_string())
                .filter(|b| !b.is_empty()),
        },
    }
}

/// Availability of every provider this binary knows how to register.
///
/// # Errors
/// If the loaded runtime rejects an availability query.
pub fn provider_statuses() -> Result<Vec<ProviderStatus>> {
    let checks: [(Provider, bool); 4] = [
        (Provider::Cpu, CPU::default().is_available()?),
        (Provider::Nnapi, NNAPI::default().is_available()?),
        (Provider::Xnnpack, XNNPACK::default().is_available()?),
        (Provider::Webgpu, WebGPU::default().is_available()?),
    ];
    Ok(checks
        .into_iter()
        .map(|(provider, available)| ProviderStatus {
            provider,
            available,
        })
        .collect())
}

/// Devices ONNX Runtime enumerates. Empty on a runtime built without device discovery, which
/// is not an error -- the rest of the report is still valid.
fn runtime_devices() -> Vec<RuntimeDevice> {
    let Ok(environment) = ort::environment::Environment::current() else {
        return Vec::new();
    };
    environment
        .devices()
        .map(|device| RuntimeDevice {
            kind: format!("{:?}", device.ty()),
            id: device.id(),
            execution_provider: device.ep().ok().map(ToString::to_string),
            vendor: device.vendor().ok().map(ToString::to_string),
        })
        .collect()
}

/// Renders a provider the way it is spelled on the command line, so anything printed can be
/// pasted straight back into `--provider`.
#[must_use]
pub fn provider_name(provider: Provider) -> String {
    format!("{provider:?}").to_lowercase()
}
