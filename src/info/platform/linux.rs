//! Linux device identity, read from sysfs.
//!
//! Desktop and server Linux publishes machine identity through DMI; single-board computers
//! (Raspberry Pi and friends) have no DMI and instead expose a device-tree model string. Both
//! are tried, because the aarch64 target exists precisely to run on the latter.

use std::path::Path;

use super::{DeviceIdentity, DeviceProbe, Fact};

pub struct LinuxProbe;

/// Reads a sysfs value, trimming the trailing newline and any interior NUL.
///
/// Device-tree strings are NUL-terminated in the file itself, which would otherwise end up
/// embedded in the report output.
fn read_sysfs(path: &str) -> Option<String> {
    let raw = std::fs::read_to_string(Path::new(path)).ok()?;
    let cleaned = raw.trim().trim_end_matches('\0').trim().to_string();
    (!cleaned.is_empty()).then_some(cleaned)
}

impl DeviceProbe for LinuxProbe {
    fn identity(&self) -> DeviceIdentity {
        let dmi_vendor = read_sysfs("/sys/class/dmi/id/sys_vendor");
        let dmi_model = read_sysfs("/sys/class/dmi/id/product_name");
        // Raspberry Pi and other device-tree boards: a single string such as
        // "Raspberry Pi 4 Model B Rev 1.4", with no separate vendor field.
        let device_tree_model = read_sysfs("/proc/device-tree/model");

        DeviceIdentity {
            manufacturer: Fact::from_option(dmi_vendor, "no DMI vendor (not an x86 platform?)"),
            model: Fact::from_option(
                dmi_model.or_else(|| device_tree_model.clone()),
                "neither DMI product_name nor device-tree model is readable",
            ),
            // Linux has no marketing-name concept; the device-tree model already is one.
            marketing_name: match device_tree_model {
                Some(model) => Fact::Known(model),
                None => Fact::not_applicable(),
            },
            // No portable SoC identifier on Linux: DMI does not carry one, and /proc/cpuinfo's
            // "Hardware" line was removed on arm64. Reported as not-applicable rather than as a
            // failure, since nothing went wrong.
            soc: Fact::not_applicable(),
        }
    }

    fn platform(&self) -> &'static str {
        "linux"
    }
}
