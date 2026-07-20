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

use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use ort::session::{RunOptions, Session, SessionInputValue};
use serde::Serialize;

use crate::config::BenchConfig;
use crate::progress::Progress;
use crate::tensors::PreparedInput;
use crate::watchdog::Watchdog;

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
    progress: &Progress,
) -> Result<Timings> {
    let run_options = Arc::new(RunOptions::new().context("creating run options")?);
    let watchdog = arm_watchdog(&run_options, config);

    let warmup_ms = iterate(
        session,
        inputs,
        config.warmup,
        "warmup",
        &run_options,
        &watchdog,
        progress,
    )?;
    let measured_ms = iterate(
        session,
        inputs,
        config.iterations,
        "measured",
        &run_options,
        &watchdog,
        progress,
    )?;

    Ok(Timings {
        warmup_ms,
        measured_ms,
    })
}

/// A watchdog that abandons an inference exceeding the configured budget.
///
/// Terminating through `RunOptions` rather than killing the process means the overrun surfaces as
/// an ordinary error, so the failure is reported the same way any other inference failure is.
fn arm_watchdog(run_options: &Arc<RunOptions>, config: BenchConfig) -> Watchdog {
    let Some(budget) = config.iteration_timeout() else {
        return Watchdog::disabled();
    };

    let terminable = Arc::clone(run_options);
    Watchdog::arm(budget, move || {
        let _ = terminable.terminate();
    })
}

/// Runs `count` inferences, returning one duration in milliseconds per iteration.
fn iterate(
    session: &mut Session,
    inputs: &[PreparedInput],
    count: u64,
    phase: &'static str,
    run_options: &RunOptions,
    watchdog: &Watchdog,
    progress: &Progress,
) -> Result<Vec<f64>> {
    let mut samples = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    let mut drawn = progress.phase(phase, count);

    for index in 0..count {
        // Built outside the timed region: `run` consumes its inputs, so the borrowed views have
        // to be rebuilt each iteration, and that allocation is harness cost rather than
        // inference cost.
        let feeds = borrow_inputs(inputs);

        let in_flight = watchdog.iteration();
        let start = Instant::now();
        let result = session.run_with_options(feeds, run_options);
        let elapsed = start.elapsed();
        drop(in_flight);

        // Checked before the error is reported: a terminated run fails with ONNX Runtime's own
        // "terminate flag" message, which describes the mechanism rather than the cause.
        let outputs = result.map_err(|err| {
            if watchdog.fired() {
                return timed_out(phase, index, elapsed);
            }
            anyhow::Error::new(err).context(format!("{phase} iteration {index} failed"))
        })?;

        // Dropped after the clock stops. Freeing the output tensors is not part of inference,
        // and it must happen before the next call regardless -- the outputs hold the session's
        // mutable borrow.
        drop(outputs);

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        samples.push(elapsed_ms);

        // After the sample is taken, never between the clock readings: drawing is harness cost.
        drawn.advance(elapsed_ms);
    }

    Ok(samples)
}

/// The error for an inference that outlived its budget.
///
/// Names the flag that set the limit and the provider question behind most real overruns, because
/// the useful next step is either to raise the budget or to stop using an execution provider that
/// cannot run this graph.
fn timed_out(phase: &str, index: u64, elapsed: std::time::Duration) -> anyhow::Error {
    anyhow!(
        "{phase} iteration {index} exceeded its time budget and was abandoned after {:.1}s. \
         Raise or remove the limit with --iteration-timeout <seconds> (0 disables it), or try a \
         different --provider: an execution provider that cannot run this graph is the usual \
         cause.",
        elapsed.as_secs_f64()
    )
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
