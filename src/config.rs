//! The settings one benchmark run executes under, separated from how they were obtained.
//!
//! `Cli` is the argv surface: flag names, defaults, parsing. `RunConfig` is what a run actually
//! does. Splitting them means a session can be built from a config that was *generated* -- which
//! is what the planned parameter sweep needs, since it must produce many configurations from one
//! command line and cannot re-parse argv for each.
//!
//! The split also draws the line between what a sweep varies and what it holds fixed:
//! `RunConfig` is the varying part, `BenchConfig` and `tensors::SynthOptions` are held constant
//! so the measurements stay comparable.

use std::path::PathBuf;

use serde::Serialize;

use crate::cli::{Cli, ExecutionMode, GraphOptLevel, LogSeverity, Provider, Toggle};

/// Everything that changes how ONNX Runtime executes the model.
///
/// Serialisable because the JSON report records the exact configuration a measurement was taken
/// under -- a latency number without its configuration is not reproducible.
#[derive(Debug, Clone, Serialize)]
pub struct RunConfig {
    pub provider: Provider,
    pub execution_mode: ExecutionMode,
    pub graph_optimization_level: GraphOptLevel,
    pub log_severity: LogSeverity,
    pub inter_op_threads: usize,
    /// `None` means "leave ONNX Runtime at its own default", which is a third state that
    /// neither a number nor zero expresses.
    pub intra_op_threads: Option<usize>,
    pub intra_op_spinning: Option<Toggle>,
    pub inter_op_spinning: Option<Toggle>,
    pub disable_cpu_arena: bool,
    pub disable_mem_pattern: bool,
    pub optimized_model_path: Option<PathBuf>,
    pub profile: bool,
    pub profile_prefix: String,
}

/// How the measurement itself is taken.
///
/// Held fixed across a sweep: changing the iteration count mid-sweep would make the resulting
/// latencies incomparable, since the tail statistics depend on how many samples were drawn.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BenchConfig {
    /// Iterations run before the measured ones. Timed and reported, but excluded from the
    /// statistics.
    pub warmup: u64,
    /// Timed iterations the statistics are computed over.
    pub iterations: u64,
}

impl From<&Cli> for RunConfig {
    fn from(cli: &Cli) -> Self {
        Self {
            provider: cli.provider,
            execution_mode: cli.execution_mode,
            graph_optimization_level: cli.graph_optimization_level,
            log_severity: cli.log_severity,
            inter_op_threads: cli.inter_op_threads,
            intra_op_threads: cli.threads,
            intra_op_spinning: cli.intra_op_spinning,
            inter_op_spinning: cli.inter_op_spinning,
            disable_cpu_arena: cli.disable_cpu_arena,
            disable_mem_pattern: cli.disable_mem_pattern,
            optimized_model_path: cli.optimized_model_path.clone(),
            profile: cli.profile,
            profile_prefix: cli.profile_prefix.clone(),
        }
    }
}

impl From<&Cli> for BenchConfig {
    fn from(cli: &Cli) -> Self {
        Self {
            warmup: cli.warmup,
            iterations: cli.iterations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).unwrap()
    }

    #[test]
    fn a_config_carries_the_flags_that_were_passed() {
        let cli = parse(&[
            "ort_runner",
            "--model",
            "m.onnx",
            "--provider",
            "cpu",
            "--inter-op-threads",
            "4",
            "--threads",
            "2",
        ]);
        let config = RunConfig::from(&cli);

        assert_eq!(config.provider, Provider::Cpu);
        assert_eq!(config.inter_op_threads, 4);
        assert_eq!(config.intra_op_threads, Some(2));
    }

    /// An unpassed thread count must stay `None` rather than becoming a number, or the session
    /// builder would override ONNX Runtime's own default with a value nobody chose.
    #[test]
    fn an_unset_thread_count_stays_absent() {
        let config = RunConfig::from(&parse(&["ort_runner", "--model", "m.onnx"]));

        assert_eq!(config.intra_op_threads, None);
        assert_eq!(config.intra_op_spinning, None);
        assert_eq!(config.inter_op_spinning, None);
    }

    #[test]
    fn bench_config_takes_the_iteration_counts() {
        let cli = parse(&[
            "ort_runner",
            "--model",
            "m.onnx",
            "--warmup",
            "5",
            "--iterations",
            "250",
        ]);
        let config = BenchConfig::from(&cli);

        assert_eq!(config.warmup, 5);
        assert_eq!(config.iterations, 250);
    }
}
