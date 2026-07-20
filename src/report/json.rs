//! Writing the complete report to a JSON file.
//!
//! Always written, never optional. The expensive part of benchmarking is getting a device into
//! the state that produced a number; discarding the samples afterwards means paying that cost
//! again to ask a question the existing data could have answered.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::paths;
use crate::report::schema::{documentation, Documentation};
use crate::report::{BenchReport, Reporter};

/// Sibling of `ort_profiler/`, holding one report per run.
const DIRECTORY: &str = "reports";

/// Bumped whenever the report's shape changes in a way a consumer would notice.
///
/// The documentation block travels with the data, so a reader does not need this to interpret one
/// file. It exists so a tool holding a directory of reports can tell which ones share a shape.
const SCHEMA_VERSION: u32 = 1;

/// Writes JSON reports into a directory, one file per run.
#[derive(Debug)]
pub struct JsonReporter {
    directory: PathBuf,
}

/// The serialised form: documentation first, then the run's fields flattened alongside it.
#[derive(Serialize)]
struct Document<'a> {
    schema_version: u32,
    documentation: Documentation,
    #[serde(flatten)]
    run: &'a BenchReport,
}

impl JsonReporter {
    /// Writes into `reports/` beside the executable. See `crate::paths` for why that anchor.
    ///
    /// # Errors
    /// If the executable's own path cannot be determined.
    pub fn beside_executable() -> Result<Self> {
        Ok(Self {
            directory: paths::beside_executable(DIRECTORY)?,
        })
    }

    #[must_use]
    pub fn in_directory(directory: PathBuf) -> Self {
        Self { directory }
    }
}

impl Reporter for JsonReporter {
    fn report(&self, report: &BenchReport) -> Result<()> {
        fs::create_dir_all(&self.directory)
            .with_context(|| format!("creating {}", self.directory.display()))?;

        let path = self
            .directory
            .join(file_name(&report.model_path, &report.created_at));

        let document = Document {
            schema_version: SCHEMA_VERSION,
            documentation: documentation(),
            run: report,
        };
        let json = serde_json::to_string_pretty(&document).context("serialising the report")?;

        fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
        anstream::println!("\nreport: {}", path.display());
        Ok(())
    }
}

/// Builds the report's filename from the model and the run's timestamp.
///
/// Both parts are deliberate: the model name makes a directory of reports skimmable, and the
/// timestamp keeps successive runs of the same model from overwriting each other. Non-alphanumeric
/// characters are dropped from the timestamp so the name is safe on any filesystem -- a colon is
/// legal on Linux but not everywhere these files get copied.
fn file_name(model_path: &str, created_at: &str) -> String {
    let model = paths::model_stem(Path::new(model_path));

    let stamp: String = created_at
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();

    format!("{model}-{stamp}.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_a_file_after_the_model_and_time() {
        assert_eq!(
            file_name("/models/mobilenet_v3.onnx", "2026-07-19T19:04:12Z"),
            "mobilenet_v3-20260719T190412Z.json"
        );
    }

    /// A colon is legal in a Linux filename but breaks when the report is copied to a Windows
    /// share or a Confluence attachment, which is exactly where these end up.
    #[test]
    fn the_name_has_no_filesystem_hostile_characters() {
        let name = file_name("m.onnx", "2026-07-19T19:04:12.123456Z");
        assert!(
            !name.contains(':') && !name.contains('/') && !name.contains(' '),
            "{name}"
        );
    }

    #[test]
    fn a_pathless_or_odd_model_still_yields_a_name() {
        assert!(file_name("", "2026").starts_with("model-"));
        assert!(file_name("/", "2026").starts_with("model-"));
        assert_eq!(file_name("plain.onnx", "2026"), "plain-2026.json");
    }

    /// Two runs of the same model must not collide, or the earlier one is silently lost.
    #[test]
    fn successive_runs_get_distinct_names() {
        let first = file_name("m.onnx", "2026-07-19T19:04:12Z");
        let second = file_name("m.onnx", "2026-07-19T19:04:13Z");
        assert_ne!(first, second);
    }
}
