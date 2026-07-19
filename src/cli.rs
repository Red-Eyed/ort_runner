//! Command-line surface.
//!
//! Every choice-valued flag is a `ValueEnum`, so the accepted strings, the `--help` listing and
//! the parsed type all come from one enum definition. The C++ had to keep an `argparse`
//! `.choices(...)` list, a `magic_enum` round-trip and the enum itself in sync by hand; drift
//! between them was a silent bug.

use std::collections::HashMap;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use serde::Serialize;

/// Per-symbolic-name overrides for dynamic input dimensions, e.g. `{"batch": 4}`.
pub type DimOverrides = HashMap<String, i64>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Cpu,
    /// Android only.
    Nnapi,
    /// Present in the Android AAR; absent from the Linux prebuilts.
    Xnnpack,
    /// Present in the Android AAR; absent from the Linux prebuilts.
    Webgpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Fill {
    Random,
    Ones,
    Zeros,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Human,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionMode {
    Sequential,
    Parallel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GraphOptLevel {
    Disable,
    Basic,
    Extended,
    /// `ORT_ENABLE_LAYOUT`. Requires a recent ONNX Runtime; see the api-24 note in Cargo.toml.
    Layout,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogSeverity {
    Verbose,
    Info,
    Warning,
    Error,
    Fatal,
}

/// A tri-state flag: unset means "leave at ONNX Runtime's own default", which is distinct from
/// both on and off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Toggle {
    On,
    Off,
}

impl Toggle {
    #[must_use]
    pub fn is_on(self) -> bool {
        self == Toggle::On
    }
}

#[derive(Debug, Parser)]
#[allow(clippy::struct_excessive_bools)]
#[command(
    name = "ort_runner",
    about = "Loads an ONNX model, auto-generates inputs from its declared shapes/dtypes, and \
             benchmarks inference latency and peak memory usage.",
    disable_version_flag = true
)]
pub struct Cli {
    /// Path to the .onnx model file.
    //
    // Not `required = true`: --version and --list-providers are useful without a model, and
    // clap would reject them before main ever runs. Validated in `Cli::model_path()` instead.
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// Path to a .npz of named input arrays (numpy.savez); array names must match model input
    /// names. Inputs not present in the archive are synthesized. Unset synthesizes everything.
    #[arg(long)]
    pub inputs: Option<PathBuf>,

    /// Path to libonnxruntime.so. Defaults to the copy sitting beside this executable.
    #[arg(long)]
    pub ort_dylib: Option<PathBuf>,

    /// Print the ONNX Runtime version and exit. Loads the shared library, so it doubles as a
    /// check that the bundled .so is present and loadable on this device.
    #[arg(long, short = 'V')]
    pub version: bool,

    /// Print runtime environment details -- versions, the shared library in use, and which
    /// execution providers it actually supports -- then exit.
    #[arg(long)]
    pub info: bool,

    /// Print declared input/output names, shapes, and dtypes, then exit without running
    /// inference.
    #[arg(long)]
    pub list_io: bool,

    /// Untimed iterations before measurement.
    #[arg(long, default_value_t = 3)]
    pub warmup: u64,

    /// Minimum iterations per measurement epoch.
    #[arg(long, default_value_t = 1)]
    pub min_epoch_iterations: u64,

    /// Intra-op thread count; left at ONNX Runtime's own default if not passed.
    #[arg(long)]
    pub threads: Option<usize>,

    /// Inter-op thread count.
    #[arg(long, default_value_t = 1)]
    pub inter_op_threads: usize,

    /// Graph execution mode; parallel only helps multi-branch graphs with
    /// --inter-op-threads > 1.
    #[arg(long, value_enum, default_value_t = ExecutionMode::Sequential)]
    pub execution_mode: ExecutionMode,

    /// Allow intra-op threads to spin instead of sleep; left at ONNX Runtime's own default if
    /// not passed.
    #[arg(long, value_enum)]
    pub intra_op_spinning: Option<Toggle>,

    /// Allow inter-op threads to spin instead of sleep; left at ONNX Runtime's own default if
    /// not passed.
    #[arg(long, value_enum)]
    pub inter_op_spinning: Option<Toggle>,

    /// Execution provider. Availability is checked against the loaded ONNX Runtime, not
    /// assumed from the build target.
    #[arg(long, value_enum, default_value_t = Provider::Cpu)]
    pub provider: Provider,

    /// Value substituted for any dynamic/symbolic input dimension not covered by --dim.
    #[arg(long, default_value_t = 1)]
    pub default_dim: i64,

    /// Override a named symbolic input dimension, e.g. --dim batch=4 (repeatable; falls back
    /// to --default-dim for unmatched or anonymous dynamic dims).
    #[arg(long = "dim", value_parser = parse_dim_override, value_name = "NAME=VALUE")]
    pub dims: Vec<(String, i64)>,

    /// Input tensor fill strategy.
    #[arg(long, value_enum, default_value_t = Fill::Random)]
    pub fill: Fill,

    /// RNG seed used when --fill=random.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Clamp for randomly-filled integer tensors (mitigates embedding/gather
    /// index-out-of-range crashes; a mitigation, not a full fix).
    #[arg(long, default_value_t = 15)]
    pub int_fill_max: i64,

    /// Human-readable text, or a single merged JSON document.
    #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
    pub output_format: OutputFormat,

    /// Enable ONNX Runtime's built-in per-op profiler (writes a Chrome-trace-format JSON file;
    /// the path is printed after the run).
    #[arg(long)]
    pub profile: bool,

    /// File prefix passed to the profiler when --profile is set.
    #[arg(long, default_value = "ort_runner_profile")]
    pub profile_prefix: String,

    /// ONNX Runtime graph optimization level.
    #[arg(long, value_enum, default_value_t = GraphOptLevel::All)]
    pub graph_optimization_level: GraphOptLevel,

    /// Disable the CPU memory arena (relevant for accurate peak-RSS measurement, since the
    /// arena pre-allocates).
    #[arg(long)]
    pub disable_cpu_arena: bool,

    /// Disable memory pattern optimization.
    #[arg(long)]
    pub disable_mem_pattern: bool,

    /// Dump the post-optimization graph to this path.
    #[arg(long)]
    pub optimized_model_path: Option<PathBuf>,

    /// ONNX Runtime session log severity level.
    #[arg(long, value_enum, default_value_t = LogSeverity::Warning)]
    pub log_severity: LogSeverity,
}

impl Cli {
    /// The `--dim` pairs as a lookup map. Later occurrences of a repeated name win, matching
    /// how the C++ built its `std::unordered_map`.
    #[must_use]
    pub fn dim_overrides(&self) -> DimOverrides {
        self.dims.iter().cloned().collect()
    }

    /// `--model` is optional at the clap layer so the model-free modes work; every other path
    /// needs it.
    ///
    /// # Errors
    /// If `--model` was not supplied.
    pub fn model_path(&self) -> anyhow::Result<&PathBuf> {
        self.model.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--model is required (or pass --version / --list-providers)")
        })
    }
}

/// Parses one `--dim name=value` entry.
///
/// Rejects an empty name or value so `--dim =4` and `--dim batch=` fail loudly rather than
/// silently registering a nonsense override.
fn parse_dim_override(entry: &str) -> Result<(String, i64), String> {
    let (name, raw_value) = entry
        .split_once('=')
        .ok_or_else(|| format!("invalid --dim value '{entry}' (expected name=value)"))?;

    if name.is_empty() || raw_value.is_empty() {
        return Err(format!(
            "invalid --dim value '{entry}' (expected name=value)"
        ));
    }

    let value = raw_value
        .parse::<i64>()
        .map_err(|_| format!("invalid --dim value '{entry}' (value must be an integer)"))?;

    Ok((name.to_string(), value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_dim_override() {
        assert_eq!(parse_dim_override("batch=4"), Ok(("batch".into(), 4)));
    }

    #[test]
    fn accepts_a_negative_value() {
        assert_eq!(parse_dim_override("n=-1"), Ok(("n".into(), -1)));
    }

    #[test]
    fn rejects_an_entry_with_no_equals_sign() {
        assert!(parse_dim_override("batch").is_err());
    }

    #[test]
    fn rejects_an_empty_name_or_value() {
        assert!(parse_dim_override("=4").is_err());
        assert!(parse_dim_override("batch=").is_err());
    }

    #[test]
    fn rejects_a_non_integer_value() {
        assert!(parse_dim_override("batch=many").is_err());
    }

    #[test]
    fn later_duplicate_dim_names_win() {
        let cli = Cli::try_parse_from([
            "ort_runner",
            "--model",
            "m.onnx",
            "--dim",
            "n=1",
            "--dim",
            "n=2",
        ])
        .expect("should parse");
        assert_eq!(cli.dim_overrides().get("n"), Some(&2));
    }

    /// clap panics on an internally inconsistent command definition; this surfaces that at
    /// test time rather than on a user's first run.
    #[test]
    fn command_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
