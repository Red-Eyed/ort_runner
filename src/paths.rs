//! Where a run's output artifacts are written.
//!
//! Two directories are produced beside the executable -- `reports/` always, `ort_profiler/` only
//! under `--profile`. Both anchor to the executable rather than the working directory: on a device
//! the binary is pushed to a known location and run from wherever the shell happens to be, so
//! anchoring to the binary is what makes `adb pull <dir>/reports` find them. The rule lives here
//! rather than in each writer so the two cannot drift onto different anchors.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// A directory named `name` beside this executable.
///
/// Returns the path without creating it; whether the directory should exist at all is the
/// caller's decision, and `ort_profiler/` deliberately does not exist on an ordinary run.
///
/// # Errors
/// If the executable's own path cannot be determined.
pub fn beside_executable(name: &str) -> Result<PathBuf> {
    let exe = std::env::current_exe().context("locating this executable")?;
    let directory = exe
        .parent()
        .context("this executable has no parent directory")?;
    Ok(directory.join(name))
}

/// The model's file stem, for naming an artifact after the model it describes.
///
/// Falls back to `model` rather than failing: a nameless artifact is still worth writing, and a
/// benchmark that refused to record its results because a path ended in a slash would be trading a
/// real loss for a cosmetic one.
#[must_use]
pub fn model_stem(model_path: &Path) -> &str {
    model_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("model")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takes_the_stem_of_a_model_path() {
        assert_eq!(
            model_stem(Path::new("/models/mobilenet_v3.onnx")),
            "mobilenet_v3"
        );
        assert_eq!(model_stem(Path::new("plain.onnx")), "plain");
    }

    #[test]
    fn a_pathless_or_odd_model_still_yields_a_name() {
        assert_eq!(model_stem(Path::new("")), "model");
        assert_eq!(model_stem(Path::new("/")), "model");
    }

    /// The directory is named, not created: `ort_profiler/` must stay absent unless --profile
    /// asked for it.
    #[test]
    fn naming_a_directory_does_not_create_it() {
        let named = beside_executable("ort_runner_paths_test_dir").unwrap();
        assert!(!named.exists(), "{} should not exist", named.display());
        assert!(named.ends_with("ort_runner_paths_test_dir"));
    }
}
