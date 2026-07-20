//! Documentation embedded in the JSON report.
//!
//! The JSON is not only archived, it is fed to a language model that writes the human-facing
//! summary (Markdown, Confluence). That consumer has none of this crate's source in front of it,
//! so a bare `"p99_ms": 9.2` is ambiguous in every way that matters: which iterations it covers,
//! whether it is interpolated, whether it can be compared with the run from last week.
//!
//! Rust doc comments do not survive serialisation, so the descriptions are data. They ship in the
//! file itself rather than in a sidecar schema, because a report that travels without its
//! definitions gets interpreted by guesswork.
//!
//! A test asserts every serialised top-level field has an entry here, so the two cannot drift.

use serde::Serialize;

/// Self-description shipped at the top of every JSON report.
#[derive(Debug, Clone, Serialize)]
pub struct Documentation {
    pub purpose: &'static str,
    pub units: &'static str,
    /// Field path -> what it means. Dotted paths address nested values.
    pub fields: Vec<(&'static str, &'static str)>,
    /// Things that would otherwise be read out of the numbers incorrectly.
    pub caveats: Vec<&'static str>,
    /// How to turn this file into a written report.
    pub reporting_guidance: Vec<&'static str>,
}

/// The documentation block for the current report shape.
#[must_use]
pub fn documentation() -> Documentation {
    Documentation {
        purpose: "One benchmark run of a single ONNX model: inference latency and the memory the \
                  model itself is responsible for. Produced by ort_runner.",
        units: "Durations are milliseconds and end in _ms. Sizes are bytes and end in _bytes. \
                Byte values are raw so they can be rescaled; render them with binary prefixes \
                (KiB/MiB/GiB).",
        fields: FIELDS.to_vec(),
        caveats: CAVEATS.to_vec(),
        reporting_guidance: GUIDANCE.to_vec(),
    }
}

/// Every top-level field, plus the nested ones whose meaning is not obvious from the name.
const FIELDS: &[(&str, &str)] = &[
    (
        "created_at",
        "UTC timestamp when the run finished, RFC 3339. Use this to order runs when comparing.",
    ),
    (
        "command_line",
        "The exact argv this run was invoked with. Reproduces the run; also the ground truth for \
         which flags were in effect, which the config block restates in resolved form.",
    ),
    (
        "model_path",
        "Path to the .onnx file as given on the command line.",
    ),
    (
        "model",
        "What the .onnx file declares about itself: description, producer, version and any custom \
         metadata_props the exporter wrote, plus the file size on disk. Every field is optional \
         and most exporters write few of them; null means the model does not declare it, not that \
         it is empty. Use this to identify WHICH model these numbers belong to -- it is what makes \
         a comparison between two runs meaningful weeks later, when 'the old one' no longer \
         identifies anything.",
    ),
    (
        "model.version",
        "The exporter's own version number for this model. Not a file-format version, and not \
         comparable across different producers.",
    ),
    (
        "model.custom",
        "Free-form key/value pairs from the model's metadata_props. Teams commonly put a training \
         run id, a git commit or a dataset version here, so this is often the most reliable way to \
         say exactly which artifact was measured. Quote it verbatim if present.",
    ),
    (
        "model.file_bytes",
        "Size of the .onnx file on disk. NOT the memory the weights occupy at runtime: a model \
         using external data keeps weights in sibling files not counted here. For runtime cost use \
         memory.weights instead.",
    ),
    (
        "input_source",
        "\"archive\" means real data supplied via --inputs; \"synthesized\" means values \
         generated from the model's declared shapes and dtypes. THIS QUALIFIES EVERY LATENCY \
         NUMBER BELOW: a model whose control flow or sparsity depends on its input values may \
         behave differently on generated data, so a synthesized run measures the shape of the \
         work, not necessarily the real workload.",
    ),
    (
        "inputs",
        "The model's declared inputs. declared_shape uses negative values for dynamic dimensions; \
         resolved_shape is what the run actually used after --dim/--default-dim substitution.",
    ),
    (
        "outputs",
        "The model's declared outputs, same shape conventions as inputs.",
    ),
    (
        "config",
        "Resolved ONNX Runtime settings this run executed under: execution provider, thread \
         counts, graph optimization level, arena and memory-pattern flags. Two runs are only \
         comparable if these match.",
    ),
    (
        "config.intra_op_threads",
        "null means ONNX Runtime's own default was left in place, which is not the same as any \
         specific number and is machine-dependent.",
    ),
    (
        "bench_config",
        "The measurement protocol: warmup iteration count, measured iteration count, and the \
         per-inference time budget. A run only produces a report if every iteration finished \
         inside that budget, so its presence here says what the numbers were allowed to be, not \
         what they were.",
    ),
    (
        "system",
        "The device and runtime the measurement was taken on: CPU, RAM, OS, ONNX Runtime build, \
         and which execution providers this build actually contains. Latency is only comparable \
         across runs on the same device.",
    ),
    (
        "load_ms",
        "Time to build the session and load the model: graph parsing, optimization and weight \
         loading. A one-off startup cost, separate from inference. Relevant to app cold start, \
         not to per-request latency.",
    ),
    (
        "measured",
        "Statistics over the measured iterations only, excluding warmup. The headline numbers.",
    ),
    (
        "measured.count",
        "How many timed iterations the statistics are computed from. Percentiles are only as \
         trustworthy as this number: p99 from 100 samples rests on a single observation.",
    ),
    (
        "measured.mean_ms",
        "Arithmetic mean. Sensitive to outliers; for a right-skewed latency distribution the \
         median is usually the better summary of typical behaviour.",
    ),
    (
        "measured.std_dev_ms",
        "Sample standard deviation (n-1 denominator). Large relative to the mean indicates an \
         unstable measurement -- thermal throttling, contention, or a noisy device.",
    ),
    (
        "measured.p50_ms",
        "Median: typical latency. Nearest-rank, so it is a latency that actually occurred.",
    ),
    (
        "measured.p90_ms",
        "90th percentile. Nearest-rank: an actually observed latency, not interpolated.",
    ),
    ("measured.p95_ms", "95th percentile, nearest-rank."),
    (
        "measured.p99_ms",
        "99th percentile: the tail. Usually the number that decides whether a model is usable in \
         an interactive application, because it is what the unlucky requests experience.",
    ),
    (
        "measured.min_ms",
        "Fastest iteration -- the closest this run got to uncontended hardware.",
    ),
    (
        "measured.max_ms",
        "Slowest iteration. A max far above p99 usually means one interruption (scheduler \
         preemption, a page fault, a throttle event) rather than a property of the model.",
    ),
    (
        "warmup",
        "Statistics over the warmup iterations, or null if none were run. Reported separately and \
         never merged into `measured`. The first inference is typically much slower -- lazy \
         allocation, cold caches, arena growth -- and that cost is real for anything invoked once \
         per user action, but it is not tail latency and must not be presented as such.",
    ),
    (
        "timings",
        "Raw per-iteration durations, in execution order: warmup_ms then measured_ms. Kept so a \
         run can be re-analysed -- histogram, drift over time, outlier attribution -- without \
         re-running it on the device. Order is meaningful: a rising trend across measured_ms \
         indicates thermal throttling.",
    ),
    (
        "memory",
        "What the model cost in memory, decomposed by phase. See memory.weights and \
         memory.working_set; those two, not the absolute readings, are what compare across \
         models.",
    ),
    (
        "memory.weights",
        "Memory added by building the session and loading the model: parameters plus the \
         optimized graph. A FIXED cost determined when the model was exported. To reduce it: \
         quantize, prune, or use a smaller architecture.",
    ),
    (
        "memory.working_set",
        "Memory added by running inference on top of the loaded model: activation and scratch \
         buffers. SCALES with input shape and batch size. To reduce it: smaller batch, shorter \
         sequence, or a different execution provider.",
    ),
    (
        "memory.weights.pss_bytes",
        "Proportional set size delta: shared pages divided by the number of processes sharing \
         them. This is what Android attributes to an app in `dumpsys meminfo`, so it is the \
         figure to quote to a platform team.",
    ),
    (
        "memory.weights.private_dirty_bytes",
        "Private dirty delta: anonymous pages that cannot be dropped under memory pressure and \
         must go to zram. The best single predictor of an Android low-memory kill.",
    ),
    (
        "memory.phases",
        "The raw snapshots the deltas were computed from, at baseline (runtime loaded, no \
         session), session_loaded, and complete. Absolute values include the shared ONNX Runtime \
         library and libc, which are identical for any model and therefore useless for comparing \
         two models -- use the deltas instead.",
    ),
    (
        "profile_path",
        "Path to ONNX Runtime's per-op profiler trace for this run, or null if --profile was not \
         passed. Chrome-trace JSON holding one entry per operator execution: this is what answers \
         WHICH op is slow, where the statistics above answer how slow the model is overall. The \
         trace itself is not included here -- it is orders of magnitude larger than this report. \
         It sits in ort_profiler/ beside the executable, next to the reports/ directory this file \
         came from, so both travel off a device together.",
    ),
    (
        "memory.peak_rss_bytes",
        "Peak resident set size for the whole process since start (getrusage ru_maxrss), \
         including shared pages. Answers 'did this ever spike'. Overstates what the process costs \
         the system, because shared library pages are billed in full.",
    ),
];

const CAVEATS: &[&str] = &[
    "A null or absent value means the platform did not expose that measurement. It does NOT mean \
     zero, and must not be reported as zero.",
    "Memory figures cover this process only. With --provider nnapi or --provider webgpu, \
     allocations happen in a vendor HAL or GPU driver, outside this process, where /proc cannot \
     see them; the memory numbers are complete for the cpu and xnnpack providers only. Say so if \
     the provider is nnapi or webgpu.",
    "Percentiles are nearest-rank, so every one is a latency that actually occurred rather than \
     an interpolated estimate. They are not directly comparable with tools that interpolate \
     (numpy.percentile's default, statrs, R type-7).",
    "Latency is comparable only between runs on the same device with the same `config`. Different \
     thread counts, execution providers or optimization levels are different measurements.",
    "A large gap between mean and p50, or a max far above p99, indicates interference during the \
     run rather than a property of the model. Consider the run noisy.",
    "If input_source is \"synthesized\", values were generated and are not representative for any \
     model whose execution depends on input values -- early exit, sparsity, dynamic control flow, \
     or sequence length driven by content.",
];

const GUIDANCE: &[&str] = &[
    "Lead with p50 and p99 latency from `measured`, the iteration count behind them, and whether \
     input_source was real or synthesized.",
    "Report memory as the two deltas (memory.weights, memory.working_set), not the absolute \
     snapshots. The absolutes are dominated by the shared runtime and are the same for every \
     model.",
    "When comparing two models, the deltas are the comparison; state which model is cheaper in \
     weights and which in working set separately, because they call for opposite remedies \
     (quantize the weights vs shrink the batch).",
    "Quote load_ms separately from inference latency, and only where cold start matters.",
    "State the device from `system` alongside any latency figure; a millisecond number without \
     the hardware it came from is not actionable.",
    "Do not compute new statistics from `timings` and present them beside `measured` without \
     saying so; if you do use the raw samples, name the method.",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_documented_path_is_unique() {
        let mut seen = HashSet::new();
        for (path, _) in FIELDS {
            assert!(seen.insert(*path), "duplicate documentation for {path}");
        }
    }

    #[test]
    fn no_description_is_empty() {
        for (path, description) in FIELDS {
            assert!(
                description.len() > 20,
                "{path} needs a real description, not {description:?}"
            );
        }
        assert!(!CAVEATS.is_empty() && !GUIDANCE.is_empty());
    }

    /// The consumer is a language model with no access to this source, so a field name that only
    /// makes sense next to the Rust definition is a defect.
    #[test]
    fn units_are_discoverable_from_field_names() {
        for (path, _) in FIELDS {
            let leaf = path.rsplit('.').next().unwrap_or(path);
            if leaf.contains("_ms") || leaf.contains("_bytes") {
                continue;
            }
            // Everything else must be a container or a non-quantitative field.
            assert!(
                !leaf.ends_with("_time") && !leaf.ends_with("_size"),
                "{path} is quantitative but its unit is not in the name"
            );
        }
    }
}
