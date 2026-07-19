//! Android device identity, read from the system property store.
//!
//! Android exposes none of this through `/proc`; the values live in the property service, which
//! `android_system_properties` reads via bionic's `__system_property_get`. That is why this
//! module -- and its dependency -- exist only on Android targets (see the `[target.'cfg(...)']`
//! section in Cargo.toml), so a Linux or macOS build never compiles or links it.

use android_system_properties::AndroidSystemProperties;

use super::{DeviceIdentity, DeviceProbe, Fact};

pub struct AndroidProbe;

impl DeviceProbe for AndroidProbe {
    fn identity(&self) -> DeviceIdentity {
        let props = AndroidSystemProperties::new();
        let read = |key: &str| Fact::from_option(props.get(key), "property not set");

        DeviceIdentity {
            manufacturer: read("ro.product.manufacturer"),
            model: read("ro.product.model"),
            // Not a standard property -- OEMs that set it (Samsung, Xiaomi) give a friendlier
            // name than the model code, so it is reported when present and absent otherwise.
            marketing_name: read("ro.product.marketname"),
            // ro.soc.model is the documented property (Android 12+). Older devices only carry
            // the vendor-specific ro.board.platform, so that is the fallback.
            soc: match props.get("ro.soc.model") {
                Some(soc) => Fact::Known(soc),
                None => Fact::from_option(
                    props.get("ro.board.platform"),
                    "neither ro.soc.model nor ro.board.platform is set",
                ),
            },
        }
    }

    fn platform(&self) -> &'static str {
        "android"
    }
}
