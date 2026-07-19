//! What environment would a run actually happen in?
//!
//! Everything here is read from the *loaded* ONNX Runtime rather than inferred from the build
//! target. The C++ decided provider availability at compile time with `#ifdef __ANDROID__`,
//! which encoded a belief that silently went stale: a comment asserted XNNPACK was available in
//! the Linux prebuilts, but those ship no XNNPACK kernels, so the flag could never have worked
//! there. Asking the runtime cannot drift, and it stays correct when the bundled shared library
//! is swapped for a different build.

use std::path::Path;

use anyhow::Result;
use ort::ep::{CPU, NNAPI, WebGPU, XNNPACK};
use ort::execution_providers::ExecutionProvider;
use serde::Serialize;

use crate::cli::Provider;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ProviderStatus {
    pub provider: Provider,
    pub available: bool,
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
    Ok(checks.into_iter().map(|(provider, available)| ProviderStatus { provider, available }).collect())
}

/// Renders a provider the way it is spelled on the command line, so anything printed here can
/// be pasted straight back into `--provider`.
#[must_use]
pub fn provider_name(provider: Provider) -> String {
    format!("{provider:?}").to_lowercase()
}

/// Prints the human-readable `--info` report.
///
/// # Errors
/// If provider availability cannot be queried.
pub fn print(dylib_path: &Path) -> Result<()> {
    println!("ort_runner:   {}", env!("CARGO_PKG_VERSION"));
    println!("onnxruntime:  {}", ort::info());
    println!("dylib:        {}", dylib_path.display());
    println!("providers:");
    for status in provider_statuses()? {
        let mark = if status.available { "yes" } else { "no " };
        println!("  {mark}  {}", provider_name(status.provider));
    }
    Ok(())
}
