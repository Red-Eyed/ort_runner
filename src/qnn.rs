//! Whether the QNN execution provider can actually run here -- and, when it cannot, why.
//!
//! Every other provider's availability is one question, answered entirely by the loaded runtime:
//! if `GetAvailableProviders` lists it, registering it works. QNN is the exception, because two
//! further conditions sit outside the runtime:
//!
//! * **The backend library.** QNN reaches the NPU through `libQnnHtp.so`, which ships in
//!   Qualcomm's QAIRT SDK under a licence that forbids redistribution. No ONNX Runtime artifact
//!   bundles it, so it has to be put on the device separately from everything else this tool
//!   ships.
//! * **The silicon.** That library drives a Hexagon DSP, which exists only on Snapdragon. On an
//!   Exynos or Dimensity phone there is no such library to find and never will be.
//!
//! Collapsing those into one "not available" would be actively misleading: the backend case is
//! fixed by copying two files, while the silicon case cannot be fixed at all, and a user told
//! only "libQnnHtp.so not found" on an Exynos would go hunting for a file that would not help
//! them if they found it. So each condition reports its own reason, and the reason travels to
//! the error message and to `--info` rather than being flattened into a bool.
//!
//! [`verdict`] is a pure function of facts the shell below gathers, so every branch is testable
//! on a host with no Android, no Qualcomm hardware and no ONNX Runtime present.

use std::path::{Path, PathBuf};

use anyhow::Result;
use ort::ep::QNN;
use ort::execution_providers::ExecutionProvider;

use crate::info::platform::{DeviceProbe, Fact};
use crate::info::Availability;

/// The QNN backend this tool targets.
///
/// QNN also ships CPU and GPU backends, but benchmarking those measures a fallback rather than
/// the accelerator anyone reaches for QNN to use, so only the HTP (Hexagon) backend counts as
/// QNN being usable here.
pub const BACKEND_LIB: &str = "libQnnHtp.so";

/// `SoC` names that are certainly not Qualcomm, and the vendor to name in the error.
///
/// Matched anywhere in the reported string, which is `ro.soc.model` on Android 12+ and the older
/// `ro.board.platform` before that -- see the `soc` field in `info::platform::android`.
const FOREIGN_SOC_NAMES: &[(&str, &str)] = &[
    ("exynos", "Samsung Exynos"),
    ("mediatek", "MediaTek"),
    ("dimensity", "MediaTek"),
    ("helio", "MediaTek"),
    ("kirin", "HiSilicon Kirin"),
    ("tensor", "Google Tensor"),
    ("unisoc", "UNISOC"),
];

/// Board-platform codes that are certainly not Qualcomm.
///
/// Matched as a *prefix* only. These are two- and three-letter codes, and `contains` on a string
/// that short would fire inside an unrelated name -- calling a Snapdragon unsupported is the one
/// error here with no recovery, since it sends someone away from hardware that does work.
const FOREIGN_BOARD_PREFIXES: &[(&str, &str)] = &[
    ("mt", "MediaTek"),              // mt6893, mt6983
    ("s5e", "Samsung Exynos"),       // s5e9945
    ("universal", "Samsung Exynos"), // universal9820
    ("hi3", "HiSilicon Kirin"),      // hi3660
    ("gs", "Google Tensor"),         // gs101, gs201
    ("ums", "UNISOC"),               // ums9620
];

/// Whether QNN can run, carrying what the caller needs either way.
///
/// `Ready` carries the backend library that was actually found, so the session configures QNN
/// with the exact file this check accepted. Handing back only "yes" would leave the caller to
/// re-derive a path, and a path derived twice can differ from the one that was validated.
#[derive(Debug, Clone)]
pub enum Readiness {
    Ready { backend: PathBuf },
    NotReady { reason: String },
}

impl Readiness {
    /// The `--info` view of this verdict.
    #[must_use]
    pub fn availability(&self) -> Availability {
        match self {
            Readiness::Ready { .. } => Availability::Available,
            Readiness::NotReady { reason } => Availability::unavailable(reason.clone()),
        }
    }

    fn not_ready(reason: impl Into<String>) -> Self {
        Readiness::NotReady {
            reason: reason.into(),
        }
    }
}

/// Where the backend library was looked for, and whether it turned up.
#[derive(Debug, Clone, Default)]
pub struct BackendSearch {
    pub found: Option<PathBuf>,
    pub searched: Vec<PathBuf>,
}

impl BackendSearch {
    /// The searched directories, for an error message that says where to put the library.
    fn searched_display(&self) -> String {
        if self.searched.is_empty() {
            return "nowhere (no executable directory and no LD_LIBRARY_PATH)".to_string();
        }
        self.searched
            .iter()
            .map(|dir| dir.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Decides whether QNN can run, from facts gathered elsewhere.
///
/// Conditions are checked cheapest-first and the first failure wins, because the first failure
/// is the one the user has to fix before any later one becomes reachable.
#[must_use]
pub fn verdict(in_build: bool, soc: &Fact<String>, backend: &BackendSearch) -> Readiness {
    if !in_build {
        return Readiness::not_ready(
            "the loaded onnxruntime does not contain the QNN execution provider. Only the \
             android-arm64 build ships it (from the onnxruntime-android-qnn AAR); the Linux \
             prebuilts and the other Android ABIs are built without it",
        );
    }

    if let Some(vendor) = foreign_vendor(soc) {
        return Readiness::not_ready(format!(
            "QNN runs only on Qualcomm Snapdragon, because its backend drives the Hexagon DSP. \
             This device reports a {vendor} SoC ({}), which has no such DSP, so no version of \
             the QNN backend can target it. Use --provider nnapi for this device's accelerator",
            soc.display()
        ));
    }

    let Some(backend_path) = &backend.found else {
        return Readiness::not_ready(format!(
            "the QNN execution provider is present, but its backend library {BACKEND_LIB} is \
             not. QNN cannot run without it. It ships in Qualcomm's QAIRT SDK, whose licence \
             forbids redistribution, so no onnxruntime artifact bundles it and it has to be \
             copied to the device separately. Searched: {}",
            backend.searched_display()
        ));
    };

    Readiness::Ready {
        backend: backend_path.clone(),
    }
}

/// The vendor named in `soc`, when it is one that certainly cannot run QNN.
///
/// Unrecognised means `None`, not "foreign". An unknown chip is not evidence of a non-Qualcomm
/// one -- `ro.soc.model` only exists on Android 12+, and the older `ro.board.platform` spells
/// Qualcomm as `kalama` or `taro`, which no vendor list would match. Unrecognised therefore
/// falls through to the backend probe, which decides on evidence rather than on a name.
fn foreign_vendor(soc: &Fact<String>) -> Option<&'static str> {
    let Fact::Known(soc) = soc else {
        return None;
    };
    let soc = soc.to_ascii_lowercase();

    if let Some((_, vendor)) = FOREIGN_SOC_NAMES.iter().find(|(name, _)| soc.contains(name)) {
        return Some(vendor);
    }
    FOREIGN_BOARD_PREFIXES
        .iter()
        .find(|(prefix, _)| soc.starts_with(prefix))
        .map(|(_, vendor)| *vendor)
}

/// Looks for the backend library where this tool's own libraries live.
///
/// Beside the executable first, matching where `dylib::resolve` finds `libonnxruntime.so` and
/// where the Android runner pushes everything, then `LD_LIBRARY_PATH`, which is how Android's
/// linker is told about that directory in the first place.
fn find_backend() -> BackendSearch {
    let mut searched: Vec<PathBuf> = Vec::new();
    if let Some(dir) = exe_dir() {
        searched.push(dir);
    }
    searched.extend(ld_library_path());
    searched.dedup();

    let found = searched
        .iter()
        .map(|dir| dir.join(BACKEND_LIB))
        .find(|candidate| candidate.is_file());

    BackendSearch { found, searched }
}

fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
}

fn ld_library_path() -> Vec<PathBuf> {
    std::env::var_os("LD_LIBRARY_PATH")
        .map(|value| std::env::split_paths(&value).collect())
        .unwrap_or_default()
}

/// Asks the loaded runtime and this device whether QNN can run.
///
/// The probe is injected rather than constructed here, so the caller decides which platform is
/// being described and this stays drivable with a fake.
///
/// # Errors
/// If the loaded runtime rejects the availability query.
pub fn readiness(probe: &dyn DeviceProbe) -> Result<Readiness> {
    let in_build = QNN::default().is_available()?;
    if !in_build {
        // Reading system properties and walking LD_LIBRARY_PATH cannot change this answer, and
        // the first reason is the one the user needs, so the device is left unqueried.
        return Ok(verdict(false, &Fact::not_applicable(), &BackendSearch::default()));
    }
    Ok(verdict(true, &probe.identity().soc, &find_backend()))
}

/// The `--info` row for QNN.
///
/// # Errors
/// If the loaded runtime rejects the availability query.
pub fn availability(probe: &dyn DeviceProbe) -> Result<Availability> {
    Ok(readiness(probe)?.availability())
}

/// A QNN provider bound to the backend library that was verified present.
#[must_use]
pub fn execution_provider(backend: &Path) -> QNN {
    QNN::default().with_backend_path(backend.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found() -> BackendSearch {
        BackendSearch {
            found: Some(PathBuf::from("/data/local/tmp/libQnnHtp.so")),
            searched: vec![PathBuf::from("/data/local/tmp")],
        }
    }

    fn missing() -> BackendSearch {
        BackendSearch {
            found: None,
            searched: vec![PathBuf::from("/data/local/tmp")],
        }
    }

    fn reason(readiness: &Readiness) -> String {
        match readiness {
            Readiness::Ready { .. } => panic!("expected NotReady"),
            Readiness::NotReady { reason } => reason.clone(),
        }
    }

    #[test]
    fn a_runtime_without_qnn_says_so_before_looking_at_the_device() {
        let readiness = verdict(false, &Fact::Known("SM8550".into()), &found());
        assert!(reason(&readiness).contains("does not contain the QNN execution provider"));
    }

    /// The reason a non-Qualcomm phone gets must not be the missing-library one: that would send
    /// someone after a file that cannot help them.
    #[test]
    fn a_foreign_soc_is_named_rather_than_blamed_on_a_missing_library() {
        for (soc, vendor) in [
            ("s5e9945", "Samsung Exynos"),
            ("exynos2400", "Samsung Exynos"),
            ("mt6893", "MediaTek"),
            ("Dimensity 9300", "MediaTek"),
            ("gs201", "Google Tensor"),
        ] {
            let readiness = verdict(true, &Fact::Known(soc.into()), &missing());
            let reason = reason(&readiness);
            assert!(reason.contains(vendor), "{soc} should name {vendor}: {reason}");
            assert!(
                !reason.contains(BACKEND_LIB),
                "{soc} must not be blamed on a missing library: {reason}"
            );
        }
    }

    /// Qualcomm's own names must never match the foreign list -- telling someone their
    /// Snapdragon is unsupported is the one wrong answer with no recovery.
    #[test]
    fn qualcomm_socs_are_never_called_foreign() {
        for soc in ["SM8550", "SM8650", "kalama", "taro", "lahaina", "sdm845", "pineapple"] {
            assert_eq!(foreign_vendor(&Fact::Known(soc.into())), None, "{soc}");
        }
    }

    /// An unreadable `SoC` is not evidence of a foreign one, so it falls through to the evidence.
    #[test]
    fn an_unknown_soc_falls_through_to_the_backend_probe() {
        let unknown: Fact<String> = Fact::from_option(None, "property not set");

        assert!(matches!(verdict(true, &unknown, &found()), Readiness::Ready { .. }));
        assert!(reason(&verdict(true, &unknown, &missing())).contains(BACKEND_LIB));
    }

    #[test]
    fn a_missing_backend_names_the_directories_that_were_searched() {
        let readiness = verdict(true, &Fact::Known("SM8550".into()), &missing());
        let reason = reason(&readiness);

        assert!(reason.contains(BACKEND_LIB));
        assert!(reason.contains("/data/local/tmp"));
        assert!(reason.contains("QAIRT"), "should say where the library comes from");
    }

    #[test]
    fn a_ready_verdict_carries_the_backend_that_was_found() {
        let readiness = verdict(true, &Fact::Known("SM8550".into()), &found());

        match readiness {
            Readiness::Ready { backend } => {
                assert_eq!(backend, PathBuf::from("/data/local/tmp/libQnnHtp.so"));
            }
            Readiness::NotReady { reason } => panic!("expected Ready, got: {reason}"),
        }
    }

    #[test]
    fn searching_nowhere_is_reported_rather_than_shown_as_an_empty_list() {
        let nowhere = BackendSearch::default();
        assert!(reason(&verdict(true, &Fact::Known("SM8550".into()), &nowhere)).contains("nowhere"));
    }
}
