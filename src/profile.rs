//! ONNX Runtime's built-in per-op profiler: where its trace goes, and flushing it.
//!
//! Self-contained so `--profile` is one opt-in feature in one file: nothing here runs unless the
//! flag was passed, and the rest of the crate only sees an optional destination path.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ort::session::Session;

use crate::paths;

/// Sibling of `reports/`, holding one Chrome-trace file per profiled run.
const DIRECTORY: &str = "ort_profiler";

/// Creates the trace directory and returns the path prefix ONNX Runtime writes under.
///
/// The prefix is a path, not a name: ONNX Runtime resolves a relative one against the process
/// working directory, which under `adb shell` is `/` and not writable -- so a relative prefix
/// silently loses the trace on exactly the devices this tool exists to measure.
///
/// ONNX Runtime appends `_<timestamp>.json` to the prefix and creates no directories of its own,
/// which is why the directory is made here, before the session is committed.
///
/// # Errors
/// If the executable's own path cannot be determined, or the directory cannot be created.
pub fn destination(model: &Path) -> Result<PathBuf> {
    let directory = paths::beside_executable(DIRECTORY)?;
    fs::create_dir_all(&directory).with_context(|| format!("creating {}", directory.display()))?;
    Ok(directory.join(paths::model_stem(model)))
}

/// Ends profiling and returns the file ONNX Runtime actually wrote.
///
/// Called explicitly rather than left to the session's drop: `ort` documents that without this the
/// trace file is left empty, and the returned name is the only way to learn the timestamp ONNX
/// Runtime appended to the prefix.
///
/// # Errors
/// If ONNX Runtime fails to finalise the trace.
pub fn finish(session: &mut Session) -> Result<PathBuf> {
    let written = session
        .end_profiling()
        .context("ending the ONNX Runtime profiler")?;
    Ok(PathBuf::from(written))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prefix has to be absolute, or ONNX Runtime resolves it against a working directory that
    /// is unwritable on Android -- the defect this module exists to prevent.
    #[test]
    fn the_destination_is_an_absolute_path_named_after_the_model() {
        let prefix = destination(Path::new("/models/mobilenet_v3.onnx")).unwrap();

        assert!(prefix.is_absolute(), "{}", prefix.display());
        assert!(prefix.ends_with("mobilenet_v3"));
        assert_eq!(
            prefix.parent().unwrap().file_name().unwrap(),
            std::ffi::OsStr::new(DIRECTORY)
        );
    }

    #[test]
    fn the_directory_exists_once_a_destination_is_prepared() {
        let prefix = destination(Path::new("m.onnx")).unwrap();

        assert!(prefix.parent().unwrap().is_dir());
    }
}
