//! A live view of a benchmark while it runs, on stderr.
//!
//! Benchmarks are the one kind of command where silence is indistinguishable from a hang: a slow
//! model on a phone prints nothing for minutes, and the only signal that anything is wrong is the
//! per-inference timeout, which says nothing about the other ninety-nine iterations.
//!
//! Two renderings, because the destination decides which is readable. A terminal gets an
//! `indicatif` bar redrawn in place; anything else gets newline-terminated lines at milestones,
//! since a redrawn bar in a captured log is one unreadable line thousands of columns wide.
//!
//! The milestone rendering is not a fallback nobody hits -- it is the default on a device.
//! `adb shell <command>` allocates no pty, so the runner always sees a pipe there, and `indicatif`
//! draws nothing to a non-terminal by design. Android therefore gets lines, which read correctly
//! whether a person is watching or the output is going into a log.
//!
//! Everything on stderr, never stdout, so the report stays pipeable into a file or a JSON parser
//! with the display still on.

use indicatif::{ProgressBar, ProgressStyle};

use crate::cli::ProgressMode;
use crate::report::format_duration;

/// Wide enough for the longest phase name, so the two phases line up under each other.
const LABEL_WIDTH: usize = 10;

/// How many lines a milestone rendering emits over a whole phase.
const MILESTONES: u64 = 10;

const TEMPLATE: &str = "{prefix:<10} [{bar:20}] {pos:>4}/{len}  {msg}  eta {eta}";

/// How a phase is drawn, once `auto` has been resolved against the terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drawing {
    Bar,
    Lines,
}

/// The progress display a benchmark reports into.
///
/// A concrete type with an absent renderer rather than a trait with a no-op implementation: there
/// are two renderings and the third state is "draw nothing", which `Option` already expresses.
/// `disabled()` is then the same null object `Watchdog` uses.
#[derive(Debug)]
pub struct Progress {
    drawing: Option<Drawing>,
}

impl Progress {
    /// A display that draws nothing.
    #[must_use]
    pub fn disabled() -> Self {
        Self { drawing: None }
    }

    /// Resolves the requested mode against the terminal it would draw to.
    ///
    /// `stderr_is_terminal` is a parameter rather than a call to `IsTerminal`, so the resolution
    /// stays pure and testable.
    #[must_use]
    pub fn new(mode: ProgressMode, stderr_is_terminal: bool) -> Self {
        // `auto` on a pipe resolves to lines rather than to nothing, which is what gives a device
        // run a display at all: `adb shell` never presents a terminal.
        let drawing = match mode {
            ProgressMode::Off => None,
            ProgressMode::Bar => Some(Drawing::Bar),
            ProgressMode::Auto if stderr_is_terminal => Some(Drawing::Bar),
            ProgressMode::Lines | ProgressMode::Auto => Some(Drawing::Lines),
        };

        Self { drawing }
    }

    /// Begins drawing one phase. The returned handle finishes the display when dropped.
    #[must_use]
    pub fn phase(&self, label: &'static str, total: u64) -> Phase {
        // A phase with no iterations has nothing to report, and the absent renderer keeps that
        // case out of every path below.
        let drawing = if total == 0 { None } else { self.drawing };

        let renderer = match drawing {
            None => None,
            Some(Drawing::Bar) => Some(Renderer::Bar(bar(label, total))),
            Some(Drawing::Lines) => Some(Renderer::Lines {
                label,
                total,
                completed: 0,
                last_emitted: 0,
            }),
        };

        Phase { renderer }
    }
}

/// One phase being drawn.
///
/// A handle whose `Drop` finishes the display, rather than a `finish()` the caller must remember:
/// a benchmark returns early on any failed inference, and a forgotten call there would leave the
/// error message printed into the middle of a bar.
#[derive(Debug)]
pub struct Phase {
    renderer: Option<Renderer>,
}

#[derive(Debug)]
enum Renderer {
    /// Redraw throttling, width detection and the ETA are all `indicatif`'s.
    Bar(ProgressBar),
    Lines {
        label: &'static str,
        total: u64,
        completed: u64,
        /// Iterations completed as of the last line, so a milestone cannot print twice.
        last_emitted: u64,
    },
}

impl Phase {
    /// Records one finished iteration, and redraws if this update is due.
    pub fn advance(&mut self, elapsed_ms: f64) {
        match &mut self.renderer {
            None => {}
            Some(Renderer::Bar(bar)) => {
                // The latest iteration rather than a running mean: the point of watching a
                // benchmark live is to see the number move, and an average stops moving early.
                bar.set_message(format_duration(elapsed_ms));
                bar.inc(1);
            }
            Some(Renderer::Lines {
                label,
                total,
                completed,
                last_emitted,
            }) => {
                *completed += 1;
                if is_milestone(*completed, *total, *last_emitted) {
                    *last_emitted = *completed;
                    eprintln!(
                        "{label:<LABEL_WIDTH$}{completed:>4}/{total}  {}",
                        format_duration(elapsed_ms)
                    );
                }
            }
        }
    }
}

impl Drop for Phase {
    /// Leaves the finished bar on screen rather than clearing it, so the two phases stay visible
    /// above the report as a record of what ran.
    fn drop(&mut self) {
        if let Some(Renderer::Bar(bar)) = &self.renderer {
            bar.finish();
        }
    }
}

fn bar(label: &'static str, total: u64) -> ProgressBar {
    let bar = ProgressBar::new(total);
    bar.set_style(style());
    bar.set_prefix(label);
    bar
}

/// The bar's layout, falling back to `indicatif`'s default if the template is ever malformed.
///
/// A fallback rather than an unwrap: a benchmark must not die because a progress bar could not be
/// styled. The template is a constant, so the fallback is unreachable in practice -- it exists so
/// that "unreachable" does not have to be enforced by a panic.
fn style() -> ProgressStyle {
    ProgressStyle::with_template(TEMPLATE).map_or_else(
        |_| ProgressStyle::default_bar(),
        |style| style.progress_chars("\u{2588}\u{2591}\u{2591}"),
    )
}

/// Whether this iteration crosses a milestone.
///
/// The last one always does, unless it was already printed: a total that divides evenly hits its
/// final milestone exactly at the end, and must not report the same line twice.
fn is_milestone(completed: u64, total: u64, last_emitted: u64) -> bool {
    if completed >= total {
        return last_emitted < completed;
    }

    // Never zero: a three-iteration warmup would otherwise round its step down and report nothing.
    let step = (total / MILESTONES).max(1);
    completed - last_emitted >= step
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The resolution that matters on a device: `adb shell` gives the runner a pipe, so `auto`
    /// there lands on lines -- which is why the milestone rendering is the Android default rather
    /// than a fallback.
    #[test]
    fn auto_draws_a_bar_on_a_terminal_and_lines_otherwise() {
        assert_eq!(
            Progress::new(ProgressMode::Auto, true).drawing,
            Some(Drawing::Bar)
        );
        assert_eq!(
            Progress::new(ProgressMode::Auto, false).drawing,
            Some(Drawing::Lines)
        );
    }

    #[test]
    fn an_explicit_mode_ignores_the_terminal() {
        assert_eq!(
            Progress::new(ProgressMode::Lines, true).drawing,
            Some(Drawing::Lines)
        );
        assert_eq!(Progress::new(ProgressMode::Off, true).drawing, None);
        assert_eq!(Progress::disabled().drawing, None);
    }

    /// `--warmup 0` is accepted by the parser, so the zero case is reachable.
    #[test]
    fn a_phase_with_no_iterations_draws_nothing() {
        let phase = Progress::new(ProgressMode::Bar, true).phase("warmup", 0);

        assert!(phase.renderer.is_none());
    }

    #[test]
    fn milestones_land_one_per_step() {
        let mut last_emitted = 0;
        let crossings = (1..=100)
            .filter(|completed| {
                let due = is_milestone(*completed, 100, last_emitted);
                if due {
                    last_emitted = *completed;
                }
                due
            })
            .count();

        assert_eq!(crossings, usize::try_from(MILESTONES).unwrap());
    }

    /// The default warmup is three iterations; a step rounded down to zero would report nothing.
    #[test]
    fn a_short_phase_reports_every_iteration() {
        assert!(is_milestone(1, 3, 0));
        assert!(is_milestone(2, 3, 1));
        assert!(is_milestone(3, 3, 2));
    }

    #[test]
    fn the_final_line_is_not_repeated_when_it_lands_on_a_milestone() {
        assert!(!is_milestone(100, 100, 100));
    }

    /// A count that does not divide evenly leaves a remainder past the last milestone, which the
    /// completion case covers.
    #[test]
    fn a_ragged_count_still_reports_completion() {
        assert!(is_milestone(105, 105, 100));
    }
}
