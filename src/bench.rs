//! Running the model repeatedly and timing each inference.
//!
//! Deliberately not an epoch-based design. Nanobench-style harnesses run a batch of iterations
//! under one clock reading and divide, because their target is code measured in nanoseconds where
//! the clock call itself dominates. One inference here is milliseconds, so a clock read costs
//! well under a thousandth of a percent, and averaging inside a batch would destroy exactly the
//! information that matters: the slow iterations. Tail latency is the number that decides whether
//! a model is usable on a phone, and an epoch mean cannot express it.
//!
//! Warmup iterations are timed and kept rather than thrown away. They are excluded from the
//! statistics -- a cold start must not show up as tail latency -- but they are the only evidence
//! of how expensive the first inference is, which matters for anything run once per user action.

use std::time::Instant;

use anyhow::{Context, Result};
use ort::session::{Session, SessionInputValue};
use serde::Serialize;

use crate::config::BenchConfig;
use crate::tensors::PreparedInput;

/// Per-iteration timings from one benchmark run, in milliseconds.
///
/// Raw samples rather than a summary: the statistics are derived from these, and the JSON report
/// carries them verbatim so a run can be re-analysed -- histogram, outlier hunt, change point --
/// without being re-run on the device.
#[derive(Debug, Clone, Serialize)]
pub struct Timings {
    pub warmup_ms: Vec<f64>,
    pub measured_ms: Vec<f64>,
}

/// Runs warmup and measured iterations, timing each one individually.
///
/// # Errors
/// If any inference fails. A failure is fatal rather than skipped: a run with missing iterations
/// is not the benchmark that was asked for, and silently reporting the ones that succeeded would
/// bias the result towards whatever was fast enough not to fail.
pub fn run(
    session: &mut Session,
    inputs: &[PreparedInput],
    config: BenchConfig,
) -> Result<Timings> {
    let warmup_ms = iterate(session, inputs, config.warmup, "warmup")?;
    let measured_ms = iterate(session, inputs, config.iterations, "measured")?;

    Ok(Timings {
        warmup_ms,
        measured_ms,
    })
}

/// Runs `count` inferences, returning one duration in milliseconds per iteration.
fn iterate(
    session: &mut Session,
    inputs: &[PreparedInput],
    count: u64,
    phase: &str,
) -> Result<Vec<f64>> {
    let mut samples = Vec::with_capacity(usize::try_from(count).unwrap_or(0));

    for index in 0..count {
        // Built outside the timed region: `run` consumes its inputs, so the borrowed views have
        // to be rebuilt each iteration, and that allocation is harness cost rather than
        // inference cost.
        let feeds = borrow_inputs(inputs);

        let start = Instant::now();
        let outputs = session
            .run(feeds)
            .with_context(|| format!("{phase} iteration {index} failed"))?;
        let elapsed = start.elapsed();

        // Dropped after the clock stops. Freeing the output tensors is not part of inference,
        // and it must happen before the next call regardless -- the outputs hold the session's
        // mutable borrow.
        drop(outputs);

        samples.push(elapsed.as_secs_f64() * 1000.0);
    }

    Ok(samples)
}

/// Borrows every prepared input as a session feed.
///
/// Views, not clones: the tensors are built once and read by every iteration. Copying them per
/// iteration would add the input size to each measurement and turn a benchmark of the model into
/// a benchmark of memcpy.
fn borrow_inputs(inputs: &[PreparedInput]) -> Vec<(&str, SessionInputValue<'_>)> {
    inputs
        .iter()
        .map(|input| (input.name.as_str(), SessionInputValue::from(&input.value)))
        .collect()
}
