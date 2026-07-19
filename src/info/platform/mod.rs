//! Where "what machine is this?" gets answered, per platform.
//!
//! The concrete probes never appear outside this module: `probe()` picks one at compile time
//! and hands back a `dyn DeviceProbe`, so everything above depends on the trait rather than on
//! Android or Linux specifics. Supporting a new platform means adding a file and one arm in
//! `probe()` -- no existing logic changes.

use serde::Serialize;

#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "linux")]
mod linux;
// Covers the development host (macOS) and anything else. Without it, `probe()` would have no
// arm to return on the machine this is most often compiled on.
#[cfg(not(any(target_os = "android", target_os = "linux")))]
mod generic;

/// Why a device fact is missing.
///
/// A bare `None` would collapse two genuinely different situations: a desktop having no `SoC`
/// model property is normal and expected, whereas failing to read one on a phone is a problem
/// worth showing. The report can only distinguish them if the reason travels with the gap.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "absent")]
pub enum Absent {
    /// This platform has no such concept, so nothing is wrong.
    NotApplicable,
    /// A read was attempted and did not produce a value.
    Unavailable { reason: String },
}

impl Absent {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Absent::Unavailable { reason: reason.into() }
    }
}

/// A value that may legitimately be missing, carrying the reason when it is.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Fact<T> {
    Known(T),
    Unknown(Absent),
}

impl<T> Fact<T> {
    /// Lifts an `Option` from a platform lookup, attributing absence to a failed read.
    pub fn from_option(value: Option<T>, reason: &str) -> Self {
        match value {
            Some(value) => Fact::Known(value),
            None => Fact::Unknown(Absent::unavailable(reason)),
        }
    }

    #[must_use]
    pub fn not_applicable() -> Self {
        Fact::Unknown(Absent::NotApplicable)
    }
}

impl<T: std::fmt::Display> Fact<T> {
    /// Rendering for the human report. Absence is shown, never silently blank, so a missing
    /// value is visibly missing rather than looking like an empty field.
    pub fn display(&self) -> String {
        match self {
            Fact::Known(value) => value.to_string(),
            Fact::Unknown(Absent::NotApplicable) => "(n/a on this platform)".to_string(),
            Fact::Unknown(Absent::Unavailable { reason }) => format!("(unavailable: {reason})"),
        }
    }
}

/// Who made and what is this machine.
#[derive(Debug, Clone, Serialize)]
pub struct DeviceIdentity {
    /// e.g. "samsung"
    pub manufacturer: Fact<String>,
    /// e.g. "SM-S911B"
    pub model: Fact<String>,
    /// Marketing name where the platform provides one, e.g. "Galaxy S23".
    pub marketing_name: Fact<String>,
    /// System-on-chip, e.g. "SM8550". The single most useful field for interpreting a
    /// benchmark, since it identifies the CPU/GPU/NPU generation.
    pub soc: Fact<String>,
}

/// Reads machine identity from whatever the platform exposes.
///
/// # Contract
/// Implementations must be **total**: never panic and never fail as a whole. A fact the
/// platform does not expose is reported as [`Absent`], because a partial answer is still
/// useful and a missing device name must not abort a benchmark run.
pub trait DeviceProbe {
    fn identity(&self) -> DeviceIdentity;

    /// Which probe produced these values, so a surprising report can be traced to its source.
    fn platform(&self) -> &'static str;
}

/// The probe for the platform this binary was built for.
#[must_use]
pub fn probe() -> Box<dyn DeviceProbe> {
    #[cfg(target_os = "android")]
    return Box::new(android::AndroidProbe);
    #[cfg(target_os = "linux")]
    return Box::new(linux::LinuxProbe);
    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    return Box::new(generic::GenericProbe);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_fact_displays_its_value() {
        assert_eq!(Fact::Known("SM8550").display(), "SM8550");
    }

    #[test]
    fn absence_is_visible_rather_than_blank() {
        let not_applicable: Fact<String> = Fact::not_applicable();
        assert_eq!(not_applicable.display(), "(n/a on this platform)");

        let failed: Fact<String> = Fact::from_option(None, "property not set");
        assert_eq!(failed.display(), "(unavailable: property not set)");
    }

    /// The trait promises totality; this pins that the built-in probe honours it.
    #[test]
    fn the_platform_probe_always_produces_an_identity() {
        let identity = probe().identity();
        // Every field must render, whether known or absent.
        assert!(!identity.soc.display().is_empty());
        assert!(!probe().platform().is_empty());
    }
}
