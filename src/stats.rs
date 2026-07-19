//! Summary statistics over a set of measured latencies.
//!
//! Every value is in milliseconds. Latency distributions are strongly right-skewed -- a few
//! iterations land far above the median because of scheduler preemption, thermal throttling or a
//! memory-allocation stall -- so the mean alone describes them badly and the high percentiles
//! are what a phone actually delivers.

use serde::Serialize;
use statrs::statistics::Statistics;

/// Summary of one set of latency samples, in milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    pub count: usize,
    pub mean_ms: f64,
    /// Sample standard deviation (n-1), which is what statrs computes.
    pub std_dev_ms: f64,
    pub min_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

/// Summarises measured latencies.
///
/// Returns `None` for an empty sample rather than a struct full of `NaN`: "there is no
/// measurement" is a different statement from "the measurement is not a number", and only the
/// first is true. statrs returns NaN here, which would propagate silently into the report.
#[must_use]
pub fn summarize(samples: &[f64]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }

    // total_cmp rather than partial_cmp: f64 is not Ord, and a NaN sample would otherwise make
    // the ordering inconsistent and the percentiles meaningless. Timings should never be NaN,
    // but the sort must be total regardless of what it is handed.
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);

    Some(Summary {
        count: sorted.len(),
        mean_ms: sorted.as_slice().mean(),
        std_dev_ms: sorted.as_slice().std_dev(),
        // Read off the sorted data rather than via statrs's min/max, which rescan the slice.
        min_ms: sorted[0],
        p50_ms: percentile(&sorted, 50.0),
        p90_ms: percentile(&sorted, 90.0),
        p95_ms: percentile(&sorted, 95.0),
        p99_ms: percentile(&sorted, 99.0),
        max_ms: sorted[sorted.len() - 1],
    })
}

/// The nearest-rank percentile of an ascending-sorted, non-empty slice.
///
/// Deliberately *not* statrs's `percentile`. That one interpolates between neighbouring order
/// statistics (the R-8 estimator), so the p50 of ten samples 1..10 is 5.5 -- a latency that
/// never occurred. Nearest-rank always returns a real observation, which is what makes "99% of
/// iterations finished within this long" a literally true statement about the run and lets the
/// number be compared against HdrHistogram-style tooling, which reports the same way.
///
/// Interpolation is the better estimator of an underlying population quantile; this tool is not
/// estimating a population, it is describing the iterations it actually ran.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    // Both casts are safe by construction: a sample count large enough to lose f64 precision
    // (2^53) is unreachable, and `rank` is clamped into [1, len] before narrowing.
    #[allow(clippy::cast_precision_loss)]
    let position = p / 100.0 * sorted.len() as f64;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rank = position.ceil().max(1.0) as usize;

    sorted[rank.min(sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1..=10, deliberately unsorted on the way in.
    fn ten() -> Vec<f64> {
        vec![1.0, 5.0, 3.0, 4.0, 10.0, 9.0, 6.0, 7.0, 8.0, 2.0]
    }

    #[test]
    fn an_empty_sample_has_no_summary() {
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn a_single_sample_is_its_own_everything() {
        let summary = summarize(&[4.0]).unwrap();

        assert_eq!(summary.count, 1);
        assert_eq!(summary.mean_ms, 4.0);
        assert_eq!(summary.min_ms, 4.0);
        assert_eq!(summary.p99_ms, 4.0);
        assert_eq!(summary.max_ms, 4.0);
    }

    #[test]
    fn input_order_does_not_affect_the_result() {
        let ascending: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(summarize(&ten()).unwrap(), summarize(&ascending).unwrap());
    }

    #[test]
    fn min_mean_and_max_are_exact() {
        let summary = summarize(&ten()).unwrap();

        assert_eq!(summary.min_ms, 1.0);
        assert_eq!(summary.max_ms, 10.0);
        assert!((summary.mean_ms - 5.5).abs() < 1e-12, "{}", summary.mean_ms);
    }

    /// The distinguishing property of nearest-rank: every percentile is a value that was
    /// actually measured. Interpolation would report 5.5 here, which no iteration took.
    #[test]
    fn percentiles_are_values_that_actually_occurred() {
        let samples = ten();
        let summary = summarize(&samples).unwrap();

        for value in [
            summary.p50_ms,
            summary.p90_ms,
            summary.p95_ms,
            summary.p99_ms,
        ] {
            assert!(
                samples.contains(&value),
                "{value} is not one of the measured samples"
            );
        }
        assert_eq!(summary.p50_ms, 5.0);
    }

    /// With 100 samples the ranks land exactly, so the percentiles are unambiguous.
    #[test]
    fn percentiles_pick_the_expected_ranks() {
        let samples: Vec<f64> = (1..=100).map(f64::from).collect();
        let summary = summarize(&samples).unwrap();

        assert_eq!(summary.p50_ms, 50.0);
        assert_eq!(summary.p90_ms, 90.0);
        assert_eq!(summary.p95_ms, 95.0);
        assert_eq!(summary.p99_ms, 99.0);
        assert_eq!(summary.max_ms, 100.0);
    }

    /// statrs uses the n-1 denominator; the population value here would be 2.0.
    #[test]
    fn the_standard_deviation_is_the_sample_one() {
        let summary = summarize(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();

        assert!(
            (summary.std_dev_ms - 2.138_089_935_299_395).abs() < 1e-9,
            "{}",
            summary.std_dev_ms
        );
    }

    /// A right-skewed sample is the realistic shape: one slow iteration must move the tail
    /// without dragging the median with it.
    #[test]
    fn an_outlier_moves_the_tail_and_not_the_median() {
        let mut samples: Vec<f64> = (1..=99).map(|_| 8.0).collect();
        samples.push(400.0);
        let summary = summarize(&samples).unwrap();

        assert_eq!(summary.p50_ms, 8.0);
        assert_eq!(summary.max_ms, 400.0);
        assert!(summary.mean_ms > 8.0, "mean should feel the outlier");
    }
}
