//! Turning parsed CLI options into a configured ONNX Runtime session.

use anyhow::{anyhow, Context, Result};
use ort::ep::{WebGPU, CPU, NNAPI, XNNPACK};
use ort::execution_providers::{ExecutionProvider, ExecutionProviderDispatch};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;

use crate::cli::{Cli, GraphOptLevel, LogSeverity, Provider};

/// Discards a session-builder error's recovery handle.
///
/// `ort` returns `Error<SessionBuilder>` from builder methods so a caller can recover the
/// builder and retry. Holding the builder makes the error neither `Send` nor `Sync`, which
/// `anyhow` requires; dropping it keeps the error code and message, which is all a CLI needs.
fn plain(err: ort::Error<SessionBuilder>) -> ort::Error {
    ort::Error::from(err)
}

/// Builds and loads the session. Timing this call is what the report calls "load time".
///
/// # Errors
/// If the requested execution provider is not present in the loaded ONNX Runtime, or the model
/// cannot be loaded.
pub fn build(cli: &Cli) -> Result<Session> {
    let model = cli.model_path()?;

    let mut builder = Session::builder()
        .context("creating a session builder")?
        .with_inter_threads(cli.inter_op_threads)
        .map_err(plain)?
        .with_parallel_execution(cli.execution_mode == crate::cli::ExecutionMode::Parallel)
        .map_err(plain)?
        .with_optimization_level(optimization_level(cli.graph_optimization_level))
        .map_err(plain)?
        .with_log_level(log_level(cli.log_severity))
        .map_err(plain)?
        .with_execution_providers(providers(cli)?)
        .map_err(plain)?;

    // Each of these means "leave ONNX Runtime at its own default" when unset, which is a third
    // state that neither `true` nor `false` expresses -- hence applying them conditionally
    // rather than passing a defaulted value.
    if let Some(threads) = cli.threads {
        builder = builder.with_intra_threads(threads).map_err(plain)?;
    }
    if let Some(spinning) = cli.intra_op_spinning {
        builder = builder
            .with_intra_op_spinning(spinning.is_on())
            .map_err(plain)?;
    }
    if let Some(spinning) = cli.inter_op_spinning {
        builder = builder
            .with_inter_op_spinning(spinning.is_on())
            .map_err(plain)?;
    }

    // ONNX Runtime's flag is "enable memory pattern"; the CLI exposes the negation because
    // disabling is the unusual, deliberate act.
    if cli.disable_mem_pattern {
        builder = builder.with_memory_pattern(false).map_err(plain)?;
    }
    if let Some(path) = &cli.optimized_model_path {
        builder = builder.with_optimized_model_path(path).map_err(plain)?;
    }
    if cli.profile {
        builder = builder.with_profiling(&cli.profile_prefix).map_err(plain)?;
    }

    builder
        .commit_from_file(model)
        .with_context(|| format!("loading model {}", model.display()))
}

/// Execution providers to register, most preferred first.
///
/// The CPU provider is always appended last as the fallback for operators the preferred
/// provider cannot handle -- which is also where `--disable-cpu-arena` has to be applied, since
/// the arena belongs to the CPU provider rather than to the session.
fn providers(cli: &Cli) -> Result<Vec<ExecutionProviderDispatch>> {
    let cpu = CPU::default()
        .with_arena_allocator(!cli.disable_cpu_arena)
        .build();

    let preferred = match cli.provider {
        Provider::Cpu => return Ok(vec![cpu]),
        Provider::Nnapi => require_available(Provider::Nnapi, NNAPI::default())?,
        Provider::Xnnpack => require_available(Provider::Xnnpack, XNNPACK::default())?,
        Provider::Webgpu => require_available(Provider::Webgpu, WebGPU::default())?,
    };

    Ok(vec![preferred, cpu])
}

/// Rejects a provider the loaded runtime does not actually contain.
///
/// The C++ decided this at compile time (`#ifdef __ANDROID__`), which encoded a belief that
/// went stale without anyone noticing: a comment asserted XNNPACK was available in the Linux
/// prebuilts, but those ship no XNNPACK kernels, so the flag could never have worked there.
/// Asking the runtime cannot drift, and it stays correct if the bundled `.so` is swapped.
fn require_available<E>(provider: Provider, ep: E) -> Result<ExecutionProviderDispatch>
where
    E: ExecutionProvider + Into<ExecutionProviderDispatch>,
{
    if !ep.is_available()? {
        return Err(anyhow!(
            "execution provider '{}' is not available in the loaded onnxruntime. \
             Run --info to see what this build supports.",
            crate::info::provider_name(provider)
        ));
    }
    Ok(ep.into())
}

fn optimization_level(level: GraphOptLevel) -> GraphOptimizationLevel {
    // Note the crate's naming: `Level3` is ORT_ENABLE_LAYOUT, while `All` is ORT_ENABLE_ALL.
    // Mapping `layout` to `All` here would silently change what the flag does.
    match level {
        GraphOptLevel::Disable => GraphOptimizationLevel::Disable,
        GraphOptLevel::Basic => GraphOptimizationLevel::Level1,
        GraphOptLevel::Extended => GraphOptimizationLevel::Level2,
        GraphOptLevel::Layout => GraphOptimizationLevel::Level3,
        GraphOptLevel::All => GraphOptimizationLevel::All,
    }
}

fn log_level(severity: LogSeverity) -> ort::logging::LogLevel {
    match severity {
        LogSeverity::Verbose => ort::logging::LogLevel::Verbose,
        LogSeverity::Info => ort::logging::LogLevel::Info,
        LogSeverity::Warning => ort::logging::LogLevel::Warning,
        LogSeverity::Error => ort::logging::LogLevel::Error,
        LogSeverity::Fatal => ort::logging::LogLevel::Fatal,
    }
}
