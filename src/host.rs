//! Process memory measurements taken alongside the inference timings.
//!
//! Latency is only half of what makes a model deployable on a device; the other half is whether
//! it fits. The question this module exists to answer is usually comparative -- two versions of
//! one model, which costs less -- and that shapes what it measures.
//!
//! Both versions run on the same ONNX Runtime, the same libc, the same everything. Those shared
//! pages are an identical constant in both measurements, so they cannot distinguish the models;
//! they only inflate both figures. What differs is *private* memory: the weights the runtime
//! allocates and the activation buffers inference needs. So the meaningful quantity is a delta
//! from a baseline, not an absolute reading, and it is worth splitting in two:
//!
//!   * after session build, minus baseline -- weights and the optimized graph, a fixed cost
//!   * peak after inference, minus after-build -- the working set, which scales with shape
//!
//! A single total hides which of those two a model is expensive in, and they call for opposite
//! fixes: quantize the weights, or shrink the batch.
//!
//! # What this cannot see
//! With `--provider nnapi` or `--provider webgpu`, allocations happen in a vendor HAL or GPU
//! driver, outside this process and outside `/proc/self/*`. These figures are complete for the
//! CPU and XNNPACK providers only.

use std::fmt;

use serde::Serialize;

use crate::info::platform::Fact;

/// Bytes per unit of `ru_maxrss`.
///
/// The field is kibibytes on Linux and bionic but bytes on Darwin. No macOS target is shipped --
/// the binary is built for Linux and Android only -- but `cargo test` can be run on a mac host,
/// and treating bytes as kibibytes would overstate memory by 1024x, which is the kind of wrong
/// that looks plausible in a report.
#[cfg(target_vendor = "apple")]
const BYTES_PER_UNIT: u64 = 1;
#[cfg(not(target_vendor = "apple"))]
const BYTES_PER_UNIT: u64 = 1024;

/// Where the kernel exposes a whole-process rollup of the per-mapping memory statistics.
///
/// Deliberately `smaps_rollup` and not `smaps`: the latter emits one block per mapping and is
/// O(mappings) to produce, which is slow enough for the kernel to be measurably perturbed by
/// being asked during a benchmark. The rollup is a single pre-summed block.
const SMAPS_ROLLUP: &str = "/proc/self/smaps_rollup";

/// One reading of this process's memory, in bytes.
///
/// Every field is a `Fact` because the whole file is Linux-specific: on a platform without it,
/// absence is reported with its reason rather than a zero that would read as "used no memory".
#[derive(Debug, Clone, Serialize)]
pub struct MemorySnapshot {
    /// Resident set size. Counts shared pages in full, so it overstates what this process costs
    /// the system -- kept because it is the figure most tools report and the easiest to compare
    /// against them.
    pub rss_bytes: Fact<u64>,
    /// Proportional set size: shared pages divided by the number of processes sharing them.
    /// This is what Android attributes to an app in `dumpsys meminfo`.
    pub pss_bytes: Fact<u64>,
    /// Private dirty pages -- anonymous memory that cannot simply be dropped under pressure and
    /// must go to zram. The best single predictor of an Android low-memory kill, and where a
    /// model's weights land when the runtime allocates them itself.
    pub private_dirty_bytes: Fact<u64>,
    /// Private clean pages, which include weights mapped from a file rather than allocated.
    /// Recorded separately because a model using external data would otherwise look cheaper
    /// than it is: those pages are real memory, just evictable.
    pub private_clean_bytes: Fact<u64>,
    /// Pages pushed out to swap, which on Android means zram. A model that only fits by
    /// swapping does not fit.
    pub swap_bytes: Fact<u64>,
}

/// The phase of a run a [`MemorySnapshot`] was taken at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// After ONNX Runtime is loaded but before a session exists. The constant both models pay.
    Baseline,
    /// After the session is built and the model loaded: weights plus the optimized graph.
    SessionLoaded,
    /// After every iteration has run: the high-water mark including activation buffers.
    Complete,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Phase::Baseline => "baseline",
            Phase::SessionLoaded => "session loaded",
            Phase::Complete => "complete",
        };
        f.write_str(text)
    }
}

/// A snapshot paired with the phase it was taken at.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseMemory {
    pub phase: Phase,
    #[serde(flatten)]
    pub snapshot: MemorySnapshot,
}

/// Reads this process's current memory usage.
#[must_use]
pub fn snapshot() -> MemorySnapshot {
    let rollup = std::fs::read_to_string(SMAPS_ROLLUP).ok();
    let reason = "/proc/self/smaps_rollup is not readable on this platform";

    let field = |key: &str| match rollup.as_deref() {
        Some(text) => Fact::from_option(smaps_field(text, key), "field absent from smaps_rollup"),
        None => Fact::from_option(None, reason),
    };

    MemorySnapshot {
        rss_bytes: field("Rss"),
        pss_bytes: field("Pss"),
        private_dirty_bytes: field("Private_Dirty"),
        private_clean_bytes: field("Private_Clean"),
        swap_bytes: field("Swap"),
    }
}

/// Reads one `Key:  <n> kB` line out of an smaps-style block, returning bytes.
///
/// Pure and text-only, so the parsing is testable against a captured sample on any platform --
/// including the macOS build host, which has no `/proc` at all.
#[must_use]
pub fn smaps_field(text: &str, key: &str) -> Option<u64> {
    text.lines()
        .find_map(|line| {
            let (name, rest) = line.split_once(':')?;
            // Exact match: "Pss" must not be satisfied by "Pss_Anon" or "Pss_File".
            (name.trim() == key).then_some(rest)
        })
        .and_then(|rest| rest.split_whitespace().next()?.parse::<u64>().ok())
        // Every size in this file is reported in kibibytes.
        .and_then(|kib| kib.checked_mul(1024))
}

/// Peak resident set size since process start, in bytes.
///
/// A high-water mark that never decreases, so it covers session construction and every iteration
/// without any sampling race to lose. It answers "did this ever spike", which a snapshot taken
/// at one instant cannot.
#[must_use]
pub fn peak_rss_bytes() -> Fact<u64> {
    // SAFETY: one of the two sanctioned unsafe blocks in this crate -- see the
    // `unsafe_code = "warn"` note in Cargo.toml, and `shutdown` for the other. `getrusage`
    // fully initialises the `struct rusage` it is handed and returns 0 on success. The pointer
    // comes from a live, correctly sized and correctly aligned `MaybeUninit<rusage>` that
    // outlives the call, and `assume_init` is reached only once that success is confirmed, so
    // uninitialised memory is never read.
    #[allow(unsafe_code)]
    let usage = unsafe {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
            return Fact::from_option(None, "getrusage(RUSAGE_SELF) returned an error");
        }
        usage.assume_init()
    };

    // ru_maxrss is a signed long -- 32-bit on armv7 -- so a negative or overflowing value is
    // representable even though the kernel should never produce one. try_from rejects it rather
    // than wrapping it into a nonsense figure.
    let peak = u64::try_from(usage.ru_maxrss)
        .ok()
        .and_then(|units| units.checked_mul(BYTES_PER_UNIT));

    Fact::from_option(peak, "getrusage reported an out-of-range ru_maxrss")
}

/// Memory attributable to the model, as the difference between two phases.
///
/// Returns `None` when either side is unavailable, or when the later reading is *below* the
/// earlier one. A negative delta is not a small cost -- it means memory was released between the
/// two points, so the subtraction no longer measures what the model added, and reporting a
/// clamped zero would disguise that.
#[must_use]
pub fn delta_bytes(earlier: &Fact<u64>, later: &Fact<u64>) -> Option<u64> {
    match (earlier, later) {
        (Fact::Known(earlier), Fact::Known(later)) => later.checked_sub(*earlier),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real capture, so the parser is tested against the kernel's actual formatting -- column
    /// alignment, the trailing unit, and the `Pss_*` siblings that must not match `Pss`.
    const SAMPLE: &str = "\
aaaad04e0000-ffffd3e26000 ---p 00000000 00:00 0                          [rollup]
Rss:                 932 kB
Pss:                 518 kB
Pss_Dirty:            80 kB
Pss_Anon:             80 kB
Pss_File:            438 kB
Shared_Clean:        824 kB
Shared_Dirty:          0 kB
Private_Clean:        28 kB
Private_Dirty:        80 kB
Referenced:          932 kB
Anonymous:            80 kB
Swap:                  0 kB
";

    #[test]
    fn parses_kibibytes_into_bytes() {
        assert_eq!(smaps_field(SAMPLE, "Rss"), Some(932 * 1024));
        assert_eq!(smaps_field(SAMPLE, "Private_Dirty"), Some(80 * 1024));
        assert_eq!(smaps_field(SAMPLE, "Swap"), Some(0));
    }

    /// `Pss` must not be satisfied by `Pss_Anon`, which appears first in some kernels and would
    /// silently substitute a much smaller number.
    #[test]
    fn a_prefix_sibling_does_not_match() {
        assert_eq!(smaps_field(SAMPLE, "Pss"), Some(518 * 1024));
    }

    #[test]
    fn an_absent_field_is_none() {
        assert_eq!(smaps_field(SAMPLE, "Hugetlb"), None);
        assert_eq!(smaps_field("", "Rss"), None);
    }

    #[test]
    fn malformed_input_does_not_panic() {
        assert_eq!(smaps_field("Rss: not-a-number kB", "Rss"), None);
        assert_eq!(smaps_field("Rss", "Rss"), None);
    }

    /// The shared runtime is what makes RSS the wrong comparative number: here it is 80% higher
    /// than PSS, and that gap is identical for every model the runtime loads.
    #[test]
    fn rss_exceeds_pss_when_pages_are_shared() {
        let rss = smaps_field(SAMPLE, "Rss").unwrap();
        let pss = smaps_field(SAMPLE, "Pss").unwrap();
        assert!(rss > pss, "shared pages should make rss exceed pss");
    }

    #[test]
    fn a_delta_measures_growth_between_phases() {
        assert_eq!(
            delta_bytes(&Fact::Known(100), &Fact::Known(175)),
            Some(75),
            "the model added 75 bytes"
        );
    }

    /// Memory released between phases means the subtraction stopped measuring what the model
    /// added; that is reported as unavailable rather than as zero growth.
    #[test]
    fn a_shrinking_reading_yields_no_delta() {
        assert_eq!(delta_bytes(&Fact::Known(200), &Fact::Known(150)), None);
    }

    #[test]
    fn a_delta_needs_both_sides() {
        let missing: Fact<u64> = Fact::from_option(None, "unavailable");
        assert_eq!(delta_bytes(&missing, &Fact::Known(150)), None);
        assert_eq!(delta_bytes(&Fact::Known(150), &missing), None);
    }

    #[test]
    fn a_running_process_reports_a_peak() {
        match peak_rss_bytes() {
            Fact::Known(bytes) => assert!(bytes > 0, "peak RSS should not be zero"),
            Fact::Unknown(absent) => panic!("peak RSS unavailable: {absent:?}"),
        }
    }

    /// Sanity-check the unit conversion: a 1024x error would pass every other test here.
    #[test]
    fn the_peak_is_in_bytes_not_kibibytes() {
        let Fact::Known(bytes) = peak_rss_bytes() else {
            panic!("peak RSS unavailable");
        };
        assert!(
            bytes > 1024 * 1024,
            "{bytes} bytes is too small to be a real peak RSS -- unit conversion is likely wrong"
        );
    }
}
