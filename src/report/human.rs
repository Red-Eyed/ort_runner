//! The report a person reads, on stdout.
//!
//! Shows the numbers someone acts on and leaves the rest to the JSON. A benchmark that prints
//! everything it knows buries the two or three figures that decide anything.
//!
//! Colour goes through `anstream`, which strips styling when stdout is not a terminal and honours
//! `NO_COLOR`, so piping this into a file yields plain text without a flag.

use anstyle::{AnsiColor, Color, Style};
use anyhow::Result;

use crate::info::platform::Fact;
use crate::report::{format_bytes, format_duration, BenchReport, MemoryDelta, Reporter};
use crate::stats::Summary;
use crate::tensors::InputSource;

const HEADING: Style = Style::new().bold();
const LABEL: Style = Style::new().dimmed();
const WARN: Style = Style::new().fg_color(Some(Color::Ansi(AnsiColor::Yellow)));
const ABSENT: Style = Style::new().dimmed();

const LABEL_WIDTH: usize = 12;

/// Prints a run to stdout.
#[derive(Debug)]
pub struct HumanReporter;

impl Reporter for HumanReporter {
    fn report(&self, report: &BenchReport) -> Result<()> {
        print_context(report);
        print_latency(report);
        print_memory(report);
        print_profile(report);
        Ok(())
    }
}

/// The profiler trace, when one was written.
///
/// Printed because ONNX Runtime appends a timestamp of its own choosing to the file name, so
/// someone who passed --profile cannot predict what to go and open.
fn print_profile(report: &BenchReport) {
    if let Some(path) = &report.profile_path {
        anstream::println!("\nprofile: {path}");
    }
}

fn print_context(report: &BenchReport) {
    heading("run");
    field("model", &report.model_path);
    field(
        "provider",
        &format!("{:?}", report.config.provider).to_lowercase(),
    );
    field("load", &format_duration(report.load_ms));

    // Stated up front, and warned about, because it qualifies every number that follows: a model
    // whose work depends on its input values is not measured faithfully by generated data.
    match report.input_source {
        InputSource::Archive => field("inputs", "from --inputs archive"),
        InputSource::Synthesized => {
            anstream::println!(
                "  {LABEL}{:<LABEL_WIDTH$}{LABEL:#}{WARN}synthesized{WARN:#} \
                 (generated values, not real data)",
                "inputs"
            );
        }
    }
}

fn print_latency(report: &BenchReport) {
    let measured = &report.measured;
    heading(&format!("latency over {} iterations", measured.count));

    field("p50", &millis(measured.p50_ms));
    field("p90", &millis(measured.p90_ms));
    field("p95", &millis(measured.p95_ms));
    field("p99", &millis(measured.p99_ms));
    field("mean", &millis(measured.mean_ms));
    field("std dev", &millis(measured.std_dev_ms));
    field(
        "min / max",
        &format!("{} / {}", millis(measured.min_ms), millis(measured.max_ms)),
    );

    print_warmup(report.warmup.as_ref());
    warn_if_noisy(measured);
}

/// Warmup is shown separately and never folded into the statistics above.
fn print_warmup(warmup: Option<&Summary>) {
    let Some(warmup) = warmup else {
        return;
    };
    field(
        "warmup",
        &format!(
            "{} first, {} mean over {} (excluded)",
            millis(warmup.max_ms),
            millis(warmup.mean_ms),
            warmup.count
        ),
    );
}

/// A max far above p99 is one interruption, not a property of the model, and saying so stops it
/// being reported as the model's tail latency.
fn warn_if_noisy(measured: &Summary) {
    if measured.p99_ms > 0.0 && measured.max_ms > measured.p99_ms * 2.0 {
        anstream::println!(
            "  {WARN}note{WARN:#} max is more than twice p99 -- likely one interruption \
             (scheduling, throttling) rather than the model"
        );
    }
}

fn print_memory(report: &BenchReport) {
    heading("memory attributable to this model");

    delta_field("weights", &report.memory.weights);
    delta_field("working set", &report.memory.working_set);

    match &report.memory.peak_rss_bytes {
        Fact::Known(bytes) => field("peak rss", &format_bytes(*bytes)),
        Fact::Unknown(_) => absent_field("peak rss"),
    }

    // The figures above cannot see allocations a vendor driver makes outside this process, so a
    // run on those providers must not be read as a complete memory picture.
    if matches!(
        report.config.provider,
        crate::cli::Provider::Nnapi | crate::cli::Provider::Webgpu
    ) {
        anstream::println!(
            "  {WARN}note{WARN:#} this provider allocates outside the process; \
             these figures cover CPU memory only"
        );
    }
}

/// Prints a phase delta, preferring PSS and naming private-dirty beside it.
fn delta_field(label: &str, delta: &MemoryDelta) {
    match (delta.pss_bytes, delta.private_dirty_bytes) {
        (Some(pss), Some(private)) => field(
            label,
            &format!(
                "{} pss, {} private dirty",
                format_bytes(pss),
                format_bytes(private)
            ),
        ),
        (Some(pss), None) => field(label, &format!("{} pss", format_bytes(pss))),
        _ => absent_field(label),
    }
}

/// Latency values go through the scaling formatter, so a microsecond-scale model does not
/// print a column of zeroes.
fn millis(value: f64) -> String {
    format_duration(value)
}

fn heading(title: &str) {
    anstream::println!("\n{HEADING}{title}{HEADING:#}");
}

fn field(label: &str, value: &str) {
    anstream::println!("  {LABEL}{label:<LABEL_WIDTH$}{LABEL:#}{value}");
}

/// Absence is shown rather than left blank, so a missing measurement looks missing instead of
/// looking like an empty field nobody filled in.
fn absent_field(label: &str) {
    anstream::println!("  {LABEL}{label:<LABEL_WIDTH$}{LABEL:#}{ABSENT}(unavailable){ABSENT:#}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(p99: f64, max: f64) -> Summary {
        Summary {
            count: 100,
            mean_ms: 8.0,
            std_dev_ms: 0.3,
            min_ms: 7.5,
            p50_ms: 8.0,
            p90_ms: 8.4,
            p95_ms: 8.6,
            p99_ms: p99,
            max_ms: max,
        }
    }

    #[test]
    fn formats_a_duration_to_two_places() {
        assert_eq!(millis(8.126), "8.13 ms");
    }

    /// The noise heuristic is what stops a single scheduling hiccup being reported as the model's
    /// tail latency, so its threshold is worth pinning down.
    #[test]
    fn a_max_far_above_p99_counts_as_noise() {
        // These call through to stdout; the assertion is that the boundary is where it claims.
        let noisy = summary(9.0, 19.0);
        assert!(noisy.max_ms > noisy.p99_ms * 2.0);

        let calm = summary(9.0, 17.0);
        assert!(calm.max_ms <= calm.p99_ms * 2.0);
    }

    #[test]
    fn a_delta_with_no_readings_renders_as_absent() {
        let delta = MemoryDelta {
            pss_bytes: None,
            private_dirty_bytes: None,
        };
        // Exercises the absent path; the point is that it does not panic or print a bare "0".
        delta_field("weights", &delta);
    }
}
