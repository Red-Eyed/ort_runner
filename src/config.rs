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
    /// Where ONNX Runtime's per-op profiler writes, or `None` to leave it off.
    ///
    /// Resolved by the caller rather than derived from the `Cli` here, because naming the
    /// destination needs the executable's own location -- an IO question that belongs at the edge,
    /// not in a value a sweep generates.
    ///
    /// Not serialised: the report records `profile_path`, the file the profiler actually wrote,
    /// which carries the timestamp ONNX Runtime appended to this prefix. Recording both would put
    /// two near-identical paths in the report for a reader to choose between.
    #[serde(skip)]
    pub profile: Option<PathBuf>,
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

impl RunConfig {
    /// Resolves the settings for one run.
    ///
    /// `profile` is injected rather than read from `cli.profile`, because turning that flag into a
    /// destination requires locating the executable and creating a directory. Taking the result as
    /// an argument keeps this constructor pure, and keeps the side effect at the one call site that
    /// is already doing IO.
    #[must_use]
    pub fn new(cli: &Cli, profile: Option<PathBuf>) -> Self {
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
            profile,
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
        let config = RunConfig::new(&cli, None);

        assert_eq!(config.provider, Provider::Cpu);
        assert_eq!(config.inter_op_threads, 4);
        assert_eq!(config.intra_op_threads, Some(2));
    }

    /// An unpassed thread count must stay `None` rather than becoming a number, or the session
    /// builder would override ONNX Runtime's own default with a value nobody chose.
    #[test]
    fn an_unset_thread_count_stays_absent() {
        let config = RunConfig::new(&parse(&["ort_runner", "--model", "m.onnx"]), None);

        assert_eq!(config.intra_op_threads, None);
        assert_eq!(config.intra_op_spinning, None);
        assert_eq!(config.inter_op_spinning, None);
    }

    /// The profiler destination is the caller's to resolve, so an unprofiled run must carry no
    /// path -- and `--profile` alone must not conjure one, or the session would enable profiling
    /// against a directory nobody created.
    #[test]
    fn the_profile_destination_is_whatever_the_caller_injected() {
        let cli = parse(&["ort_runner", "--model", "m.onnx", "--profile"]);

        assert_eq!(RunConfig::new(&cli, None).profile, None);
        assert_eq!(
            RunConfig::new(&cli, Some(PathBuf::from("/tmp/ort_profiler/m"))).profile,
            Some(PathBuf::from("/tmp/ort_profiler/m"))
        );
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
