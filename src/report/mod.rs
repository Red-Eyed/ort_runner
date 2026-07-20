//! Turning a finished run into output.
//!
//! Two destinations, one payload. The human report goes to stdout and shows the handful of
//! numbers a person acts on; the JSON report goes to a file and carries everything, including the
//! raw per-iteration samples, so a run can be re-analysed later without being re-run on the
//! device. Getting a device back into the same state is the expensive part of benchmarking, so
//! throwing away the samples that were already collected would be the costly mistake.
//!
//! Both are `Reporter` implementations, so adding a destination means adding a type rather than
//! editing the ones that exist.

pub mod human;
pub mod json;
pub mod schema;

use anyhow::Result;
use serde::Serialize;

use crate::bench::Timings;
use crate::config::{BenchConfig, RunConfig};
use crate::host::{MemorySnapshot, PhaseMemory};
use crate::info::platform::Fact;
use crate::info::SystemInfo;
use crate::model::{InputSpec, ModelDescription, OutputSpec};
use crate::stats::Summary;
use crate::tensors::InputSource;

/// Somewhere a finished run can be sent.
///
/// One method, because that is all any destination needs. A reporter that also wanted to be
/// asked "are you enabled" or "what is your path" would be pushing its own wiring into the
/// caller.
pub trait Reporter {
    /// # Errors
    /// If the report cannot be rendered or written.
    fn report(&self, report: &BenchReport) -> Result<()>;
}

/// Memory growth between two phases, in bytes.
///
/// Both metrics are kept because they answer different questions. PSS is what Android attributes
/// to the app; private-dirty is what cannot be evicted under pressure. A model whose weights are
/// mapped from an external data file shows up in one and not the other, and that difference is
/// the information, not noise to be averaged away.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryDelta {
    pub pss_bytes: Option<u64>,
    pub private_dirty_bytes: Option<u64>,
}

impl MemoryDelta {
    fn between(earlier: &MemorySnapshot, later: &MemorySnapshot) -> Self {
        Self {
            pss_bytes: crate::host::delta_bytes(&earlier.pss_bytes, &later.pss_bytes),
            private_dirty_bytes: crate::host::delta_bytes(
                &earlier.private_dirty_bytes,
                &later.private_dirty_bytes,
            ),
        }
    }
}

/// What the run cost in memory, decomposed.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryReport {
    /// Every raw snapshot, so the JSON keeps what the deltas were derived from.
    pub phases: Vec<PhaseMemory>,
    /// Session build minus baseline: weights and the optimized graph. A fixed cost, decided when
    /// the model was exported.
    pub weights: MemoryDelta,
    /// Run complete minus session built: the inference working set, which scales with the input
    /// shapes rather than with the model's parameter count.
    pub working_set: MemoryDelta,
    pub peak_rss_bytes: Fact<u64>,
}

impl MemoryReport {
    /// Builds the decomposition from the three phase snapshots.
    #[must_use]
    pub fn new(
        baseline: MemorySnapshot,
        session_loaded: MemorySnapshot,
        complete: MemorySnapshot,
        peak_rss_bytes: Fact<u64>,
    ) -> Self {
        let weights = MemoryDelta::between(&baseline, &session_loaded);
        let working_set = MemoryDelta::between(&session_loaded, &complete);

        Self {
            phases: vec![
                PhaseMemory {
                    phase: crate::host::Phase::Baseline,
                    snapshot: baseline,
                },
                PhaseMemory {
                    phase: crate::host::Phase::SessionLoaded,
                    snapshot: session_loaded,
                },
                PhaseMemory {
                    phase: crate::host::Phase::Complete,
                    snapshot: complete,
                },
            ],
            weights,
            working_set,
            peak_rss_bytes,
        }
    }
}

/// Everything one benchmark run produced.
///
/// Deliberately complete rather than minimal: it carries the command line, the resolved
/// configuration, the device it ran on and every raw sample. A latency figure without the
/// configuration that produced it is not reproducible, and on-device numbers are compared across
/// weeks and devices where nobody remembers which flags were passed.
#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub created_at: String,
    pub command_line: Vec<String>,
    pub model_path: String,
    /// What the .onnx file says about itself. Identifies *which* model produced these numbers,
    /// which is what makes a comparison between two runs meaningful later.
    pub model: ModelDescription,
    /// Whether the run measured real data or generated data -- the first thing that qualifies
    /// every number below it.
    pub input_source: InputSource,
    pub inputs: Vec<InputSpec>,
    pub outputs: Vec<OutputSpec>,
    pub config: RunConfig,
    pub bench_config: BenchConfig,
    pub system: SystemInfo,
    /// Session construction and model load, in milliseconds.
    pub load_ms: f64,
    /// Statistics over the measured iterations only, never the warmup.
    pub measured: Summary,
    /// Warmup is summarised separately rather than merged, so a cold first inference is visible
    /// as itself instead of as tail latency.
    pub warmup: Option<Summary>,
    pub timings: Timings,
    pub memory: MemoryReport,
    /// The per-op profiler trace this run wrote, or `None` if `--profile` was not passed.
    ///
    /// An artifact rather than a setting, which is why it sits here and not in `config`: the name
    /// carries a timestamp ONNX Runtime chose, so it is only knowable after the run.
    pub profile_path: Option<String>,
}

/// Binary-prefix size, scaled to the unit that keeps the number readable.
///
/// Model memory spans kilobytes to gigabytes, so a fixed unit is wrong at one end or the other:
/// GiB renders a 40 MiB working set as "0.0 GiB".
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;

    #[allow(clippy::cast_precision_loss)]
    let value = bytes as f64;

    if value >= GIB {
        format!("{:.2} GiB", value / GIB)
    } else if value >= MIB {
        format!("{:.1} MiB", value / MIB)
    } else if value >= KIB {
        format!("{:.1} KiB", value / KIB)
    } else {
        format!("{bytes} B")
    }
}

/// A duration in milliseconds, scaled to the unit that keeps it readable.
///
/// Inference spans six orders of magnitude across the models this tool is pointed at: a fused
/// elementwise graph finishes in microseconds while a transformer takes hundreds of milliseconds.
/// A fixed `ms` rendering prints the fast end as "0.00 ms", which reads as a broken measurement
/// rather than a fast one.
///
/// The JSON keeps raw milliseconds regardless; this is presentation only.
#[must_use]
pub fn format_duration(ms: f64) -> String {
    const MS_PER_SECOND: f64 = 1000.0;
    const US_PER_MS: f64 = 1000.0;
    const NS_PER_MS: f64 = 1_000_000.0;

    if ms >= MS_PER_SECOND {
        format!("{:.2} s", ms / MS_PER_SECOND)
    } else if ms >= 1.0 {
        format!("{ms:.2} ms")
    } else if ms >= 0.001 {
        format!("{:.1} µs", ms * US_PER_MS)
    } else {
        format!("{:.0} ns", ms * NS_PER_MS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_to_a_readable_unit() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.0 KiB");
        assert_eq!(format_bytes(40 * 1024 * 1024), "40.0 MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.00 GiB");
    }

    /// The reason for scaling at all: a fixed GiB unit renders a realistic working set as zero.
    #[test]
    fn a_working_set_does_not_render_as_zero() {
        assert_eq!(format_bytes(45 * 1024 * 1024), "45.0 MiB");
    }

    #[test]
    fn durations_scale_to_a_readable_unit() {
        assert_eq!(format_duration(1500.0), "1.50 s");
        assert_eq!(format_duration(8.126), "8.13 ms");
        assert_eq!(format_duration(0.002_875), "2.9 µs");
        assert_eq!(format_duration(0.000_4), "400 ns");
    }

    /// The defect this exists to prevent: a real microsecond-scale inference rendered as
    /// "0.00 ms", which reads as a broken measurement rather than a fast one.
    #[test]
    fn a_microsecond_inference_does_not_render_as_zero() {
        let rendered = format_duration(0.002_875);
        assert!(!rendered.starts_with("0.00"), "{rendered}");
    }

    fn snapshot(pss: u64, private_dirty: u64) -> MemorySnapshot {
        MemorySnapshot {
            rss_bytes: Fact::Known(pss * 2),
            pss_bytes: Fact::Known(pss),
            private_dirty_bytes: Fact::Known(private_dirty),
            private_clean_bytes: Fact::Known(0),
            swap_bytes: Fact::Known(0),
        }
    }

    #[test]
    fn a_delta_measures_growth_between_two_snapshots() {
        let delta = MemoryDelta::between(&snapshot(200, 100), &snapshot(350, 180));

        assert_eq!(delta.pss_bytes, Some(150));
        assert_eq!(delta.private_dirty_bytes, Some(80));
    }

    #[test]
    fn the_decomposition_splits_weights_from_working_set() {
        let memory = MemoryReport::new(
            snapshot(100, 50),
            snapshot(400, 300),
            snapshot(470, 340),
            Fact::Known(1024),
        );

        assert_eq!(memory.weights.pss_bytes, Some(300));
        assert_eq!(memory.working_set.pss_bytes, Some(70));
        assert_eq!(memory.phases.len(), 3);
    }

    /// The documentation ships inside every report, so it is duplicated across files and cannot be
    /// fixed in one place after the fact. That makes drift the real risk of embedding it: a field
    /// added here and not described there reaches the language model as an undocumented number.
    /// This fails the build instead.
    #[test]
    fn every_serialised_field_is_documented() {
        let report = sample_report();
        let value = serde_json::to_value(&report).unwrap();
        let object = value.as_object().unwrap();

        let documented: std::collections::HashSet<&str> = schema::documentation()
            .fields
            .iter()
            .map(|(path, _)| *path)
            .collect();

        let undocumented: Vec<&String> = object
            .keys()
            .filter(|key| !documented.contains(key.as_str()))
            .collect();

        assert!(
            undocumented.is_empty(),
            "these report fields have no entry in report::schema: {undocumented:?}"
        );
    }

    /// Guards the other direction: a documented path that no longer exists is stale advice, and
    /// tells the language model to look for something that is not there.
    #[test]
    fn no_documented_top_level_field_is_stale() {
        let value = serde_json::to_value(sample_report()).unwrap();
        let object = value.as_object().unwrap();

        for (path, _) in &schema::documentation().fields {
            // Nested paths are addressed with dots; only the top level is checked here.
            if path.contains('.') {
                continue;
            }
            assert!(
                object.contains_key(*path),
                "report::schema documents '{path}', which the report does not serialise"
            );
        }
    }

    fn sample_report() -> BenchReport {
        use crate::cli::{ExecutionMode, Fill, GraphOptLevel, LogSeverity, Provider};
        use crate::info::platform::DeviceIdentity;
        use crate::info::{CpuInfo, HostInfo, SystemInfo};

        let _ = Fill::Random;

        BenchReport {
            created_at: "2026-07-19T19:04:12Z".into(),
            command_line: vec!["ort_runner".into()],
            model_path: "m.onnx".into(),
            model: crate::model::ModelDescription::default(),
            input_source: InputSource::Synthesized,
            inputs: vec![],
            outputs: vec![],
            config: RunConfig {
                provider: Provider::Cpu,
                execution_mode: ExecutionMode::Sequential,
                graph_optimization_level: GraphOptLevel::All,
                log_severity: LogSeverity::Warning,
                inter_op_threads: 1,
                intra_op_threads: None,
                intra_op_spinning: None,
                inter_op_spinning: None,
                disable_cpu_arena: false,
                disable_mem_pattern: false,
                optimized_model_path: None,
                profile: None,
            },
            bench_config: BenchConfig {
                warmup: 3,
                iterations: 100,
            },
            system: SystemInfo {
                ort_runner_version: "0.3.0".into(),
                onnxruntime_build: "build".into(),
                dylib_path: "libonnxruntime.so".into(),
                platform: "linux".into(),
                device: DeviceIdentity {
                    manufacturer: Fact::not_applicable(),
                    model: Fact::not_applicable(),
                    marketing_name: Fact::not_applicable(),
                    soc: Fact::not_applicable(),
                },
                host: HostInfo {
                    os: None,
                    os_version: None,
                    kernel: None,
                    hostname: None,
                    total_memory_bytes: 0,
                    cpu: CpuInfo {
                        arch: "aarch64".into(),
                        brand: None,
                        logical_cores: 4,
                    },
                },
                providers: vec![],
                runtime_devices: vec![],
            },
            load_ms: 1.0,
            measured: Summary {
                count: 1,
                mean_ms: 1.0,
                std_dev_ms: 0.0,
                min_ms: 1.0,
                p50_ms: 1.0,
                p90_ms: 1.0,
                p95_ms: 1.0,
                p99_ms: 1.0,
                max_ms: 1.0,
            },
            warmup: None,
            timings: Timings {
                warmup_ms: vec![],
                measured_ms: vec![1.0],
            },
            memory: MemoryReport::new(
                snapshot(1, 1),
                snapshot(2, 2),
                snapshot(3, 3),
                Fact::Known(1),
            ),
            profile_path: None,
        }
    }
}
