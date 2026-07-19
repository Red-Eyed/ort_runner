//! Locating the ONNX Runtime shared library.
//!
//! The C++ build solved this with an `$ORIGIN`-relative rpath baked in by the linker. Under
//! `load-dynamic` nothing links against ONNX Runtime at all, so the same "ship the .so beside
//! the binary" layout is reproduced here explicitly: look next to the executable. That is
//! plain code rather than linker configuration, so it can be inspected, overridden and
//! reported in an error message.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

/// Library filenames to look for beside the executable, most specific first.
///
/// The Linux release tarball ships a versioned soname (`libonnxruntime.so.1`) while the
/// Android AAR ships a bare `libonnxruntime.so`, so both spellings are probed.
const CANDIDATES: &[&str] = &[
    "libonnxruntime.so",
    "libonnxruntime.so.1",
    "libonnxruntime.dylib",
    "onnxruntime.dll",
];

/// Resolves the shared library to load.
///
/// An explicit `--ort-dylib` wins and is validated eagerly, so a typo reports the path the user
/// typed instead of falling back to a bundled copy and hiding the mistake.
///
/// # Errors
/// If an explicit path does not exist, or no bundled library is found beside the executable.
pub fn resolve(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if !path.is_file() {
            return Err(anyhow!("--ort-dylib {} does not exist", path.display()));
        }
        return Ok(path.to_path_buf());
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .ok_or_else(|| anyhow!("could not determine this executable's directory"))?;

    beside(&exe_dir).ok_or_else(|| {
        anyhow!(
            "no ONNX Runtime shared library found next to the executable ({}); \
             expected one of {}. Pass --ort-dylib <path> to point at one.",
            exe_dir.display(),
            CANDIDATES.join(", ")
        )
    })
}

/// First candidate library present in `dir`, if any. Pure, so it is testable against a
/// scratch directory without touching the real executable path.
fn beside(dir: &Path) -> Option<PathBuf> {
    CANDIDATES
        .iter()
        .map(|name| dir.join(name))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_path_is_rejected_when_missing() {
        let err = resolve(Some(Path::new("/nonexistent/libonnxruntime.so"))).unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn finds_nothing_in_an_empty_directory() {
        let dir = std::env::temp_dir().join("ort_runner_dylib_empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(beside(&dir), None);
    }

    #[test]
    fn finds_a_candidate_that_is_present() {
        let dir = std::env::temp_dir().join("ort_runner_dylib_present");
        std::fs::create_dir_all(&dir).unwrap();
        let lib = dir.join("libonnxruntime.so");
        std::fs::write(&lib, b"not really a library").unwrap();

        assert_eq!(beside(&dir), Some(lib.clone()));

        std::fs::remove_file(&lib).unwrap();
    }
}
