//! Fallback identity for platforms this tool does not deploy to.
//!
//! `ort_runner` ships for Android and Linux; macOS only ever runs it as a development host. This
//! probe exists so the host build compiles and behaves, and so the trait's totality contract
//! holds everywhere rather than only on the deployment targets.

use super::{DeviceIdentity, DeviceProbe, Fact};

pub struct GenericProbe;

impl DeviceProbe for GenericProbe {
    fn identity(&self) -> DeviceIdentity {
        // Nothing here is a failure: this platform simply is not one whose identity we read.
        DeviceIdentity {
            manufacturer: Fact::not_applicable(),
            model: Fact::not_applicable(),
            marketing_name: Fact::not_applicable(),
            soc: Fact::not_applicable(),
        }
    }

    fn platform(&self) -> &'static str {
        std::env::consts::OS
    }
}
