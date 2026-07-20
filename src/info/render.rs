//! Rendering [`SystemInfo`] for humans and for machines.
//!
//! Both formats read the same gathered value, so they cannot drift the way two independent
//! print paths would. Colour goes through `anstream`, which strips styling automatically when
//! stdout is not a terminal or `NO_COLOR` is set -- so piping `--info` into a file or into `adb
//! shell` output stays clean without any tty detection here.

use anstyle::{AnsiColor, Color, Style};
use anyhow::Result;

use super::{provider_name, SystemInfo};
use crate::cli::OutputFormat;

const HEADING: Style = Style::new().bold();
const LABEL: Style = Style::new().dimmed();
const GOOD: Style = Style::new().fg_color(Some(Color::Ansi(AnsiColor::Green)));
const ABSENT: Style = Style::new().dimmed();

/// Width that keeps every label column aligned across all sections.
const LABEL_WIDTH: usize = 16;

/// # Errors
/// If JSON serialization fails.
pub fn render(info: &SystemInfo, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Human => {
            render_human(info);
            Ok(())
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(info)?);
            Ok(())
        }
    }
}

fn render_human(info: &SystemInfo) {
    section("ort_runner");
    field("version", &info.ort_runner_version);
    field("dylib", &info.dylib_path);
    field("onnxruntime", &info.onnxruntime_build);

    section("device");
    field("platform", &info.platform);
    field("manufacturer", &info.device.manufacturer.display());
    field("model", &info.device.model.display());
    field("marketing name", &info.device.marketing_name.display());
    field("soc", &info.device.soc.display());

    section("host");
    field(
        "os",
        &optional(info.host.os.as_deref(), info.host.os_version.as_deref()),
    );
    field("kernel", &info.host.kernel.clone().unwrap_or_else(unknown));
    field(
        "hostname",
        &info.host.hostname.clone().unwrap_or_else(unknown),
    );
    field("arch", &info.host.cpu.arch);
    field("cpu", &info.host.cpu.brand.clone().unwrap_or_else(unknown));
    field("logical cores", &info.host.cpu.logical_cores.to_string());
    field("memory", &format_bytes(info.host.total_memory_bytes));

    section("execution providers");
    for status in &info.providers {
        let name = provider_name(status.provider);
        // The reason is printed rather than summarised: a provider can be missing for reasons
        // that need opposite responses from the reader, and "no" alone cannot tell them apart.
        match status.availability.reason() {
            None => anstream::println!("  {GOOD}yes{GOOD:#}  {name}"),
            Some(reason) => anstream::println!("  {ABSENT}no   {name} ({reason}){ABSENT:#}"),
        }
    }

    section("runtime devices");
    if info.runtime_devices.is_empty() {
        anstream::println!("  {ABSENT}(none reported by this runtime){ABSENT:#}");
    }
    for device in &info.runtime_devices {
        let ep = device.execution_provider.as_deref().unwrap_or("?");
        let vendor = device.vendor.as_deref().unwrap_or("?");
        anstream::println!(
            "  {:<6} id={}  vendor={vendor}  ep={ep}",
            device.kind,
            device.id
        );
    }
}

fn section(title: &str) {
    anstream::println!("\n{HEADING}{title}{HEADING:#}");
}

fn field(label: &str, value: &str) {
    anstream::println!("  {LABEL}{label:<LABEL_WIDTH$}{LABEL:#}{value}");
}

fn unknown() -> String {
    "(unknown)".to_string()
}

/// Joins an OS name with its version when both are present, since either alone is only half an
/// answer.
fn optional(name: Option<&str>, version: Option<&str>) -> String {
    match (name, version) {
        (Some(name), Some(version)) => format!("{name} {version}"),
        (Some(name), None) => name.to_string(),
        (None, Some(version)) => version.to_string(),
        (None, None) => unknown(),
    }
}

/// Binary-prefix size, which is how device RAM is conventionally quoted.
fn format_bytes(bytes: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    format!("{gib:.1} GiB")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_memory_in_gibibytes() {
        assert_eq!(format_bytes(8 * 1024 * 1024 * 1024), "8.0 GiB");
    }

    #[test]
    fn joins_os_name_and_version_when_both_present() {
        assert_eq!(optional(Some("Debian"), Some("12")), "Debian 12");
        assert_eq!(optional(Some("Debian"), None), "Debian");
        assert_eq!(optional(None, None), "(unknown)");
    }
}
