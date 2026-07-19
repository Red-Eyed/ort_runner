//! Building the input tensors a benchmark run feeds to the model.

pub mod dtype;
pub mod load;
pub mod synth;

use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use ort::value::DynValue;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use crate::cli::Fill;
use crate::model::InputSpec;

/// Where one input's values came from.
///
/// An enum rather than a bool or a string because it is reported to the user and serialised
/// into the JSON report: "was this measured on real data or made-up data" is the first question
/// anyone reading a benchmark result should be able to answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InputSource {
    /// Read from the `--inputs` archive.
    Archive,
    /// Generated from the model's declared shape and dtype.
    Synthesized,
}

/// Knobs deciding the *values* of synthesized inputs.
///
/// Separate from the session configuration: these change what the tensors contain, not how
/// ONNX Runtime executes them, so a sweep over execution settings holds these fixed.
#[derive(Debug, Clone, Copy)]
pub struct SynthOptions {
    pub fill: Fill,
    pub seed: u64,
    pub int_max: i64,
}

/// One input, resolved and ready to feed the session.
pub struct PreparedInput {
    pub name: String,
    /// The shape actually used. For an archive input this is the file's shape, which may differ
    /// from `resolved_shape` wherever the model left a dimension dynamic.
    pub shape: Vec<i64>,
    pub value: DynValue,
}

impl std::fmt::Debug for PreparedInput {
    /// Omits the tensor itself, as `LoadedArray` does and for the same reason: these are
    /// routinely megabytes, and anything printing an input wants to know which input it is,
    /// not what is in it. `DynValue` implements no `Debug` of its own either.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedInput")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

/// Everything a run needs to call `session.run()`.
///
/// `source` is one value for the whole run, not one per input: an archive must supply every
/// input or none, so a run cannot be half real data and half generated. Keeping it here rather
/// than on each input makes that invariant structural instead of something a reader has to
/// check by scanning the inputs.
#[derive(Debug)]
pub struct PreparedInputs {
    pub inputs: Vec<PreparedInput>,
    pub source: InputSource,
}

/// How an archive fails to line up with the model. Both empty means it matches exactly.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ArchiveMismatch {
    /// Inputs the model declares that the archive does not supply.
    pub missing: Vec<String>,
    /// Arrays the archive supplies that are not inputs of this model.
    pub unused: Vec<String>,
}

impl ArchiveMismatch {
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.missing.is_empty() && self.unused.is_empty()
    }
}

impl std::fmt::Display for ArchiveMismatch {
    /// Names both sides of the mismatch. A "does not match" message that does not say *which*
    /// names are wrong leaves the user diffing `--list-io` against their savez call by hand,
    /// which is the actual work.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "the --inputs archive does not match this model")?;
        if !self.missing.is_empty() {
            writeln!(
                f,
                "  declared by the model, missing from the archive: {}",
                self.missing.join(", ")
            )?;
        }
        if !self.unused.is_empty() {
            writeln!(
                f,
                "  present in the archive, not an input of this model: {}",
                self.unused.join(", ")
            )?;
        }
        write!(
            f,
            "Every input must be supplied by the archive; ort_runner will not synthesize the \
             difference, because a benchmark run half on real data and half on generated data \
             is not a measurement of either. Run --list-io to see the model's inputs, and check \
             the keys passed to numpy.savez."
        )
    }
}

/// Compares an archive's keys against the model's declared inputs.
///
/// Pure and name-only, so the rule is testable with no archive on disk and no ONNX Runtime
/// loaded -- building the tensors themselves needs both.
#[must_use]
pub fn match_archive(specs: &[InputSpec], archive_names: &[&str]) -> ArchiveMismatch {
    let supplied: HashSet<&str> = archive_names.iter().copied().collect();
    let declared: HashSet<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();

    // Sorted, because HashMap iteration order is arbitrary and an error message that reorders
    // between runs is hard to compare against the last one.
    let mut missing: Vec<String> = specs
        .iter()
        .map(|spec| spec.name.clone())
        .filter(|name| !supplied.contains(name.as_str()))
        .collect();
    let mut unused: Vec<String> = archive_names
        .iter()
        .filter(|name| !declared.contains(*name))
        .map(|name| (*name).to_string())
        .collect();
    missing.sort();
    unused.sort();

    ArchiveMismatch { missing, unused }
}

/// Builds every input tensor the model declares.
///
/// With no `--inputs` archive every input is synthesized. With one, the archive must supply
/// every input and nothing else -- a mismatch is an error naming both sides, never a silent
/// fallback to synthesis. Passing `--inputs` is a statement that the run should use that data,
/// and quietly generating part of it would answer a question the user did not ask while
/// looking exactly like the one they did.
///
/// # Errors
/// If the archive cannot be read, does not match the model's inputs exactly, holds an array
/// disagreeing with an input's declared dtype/rank/shape, or an input cannot be synthesized.
pub fn prepare_inputs(
    specs: &[InputSpec],
    archive: Option<&Path>,
    options: SynthOptions,
) -> Result<PreparedInputs> {
    let Some(path) = archive else {
        return synthesize_all(specs, options);
    };

    let mut arrays = load::read_npz(path)?;

    // Scoped so this borrow of `arrays` ends before the loop below starts removing from it.
    let mismatch = {
        let names: Vec<&str> = arrays.keys().map(String::as_str).collect();
        match_archive(specs, &names)
    };
    if !mismatch.is_exact() {
        bail!("{mismatch}\n  archive: {}", path.display());
    }

    let inputs = specs
        .iter()
        .map(|spec| {
            // match_archive established above that every input is present, so this cannot fire;
            // it is an error rather than an expect because library code does not get to panic
            // on an invariant it believes it has proved.
            let array = arrays.remove(&spec.name).ok_or_else(|| {
                anyhow!(
                    "input '{}' missing from the archive after validation",
                    spec.name
                )
            })?;
            // Read the shape before the conversion consumes the array.
            let shape = array.shape().to_vec();
            Ok(PreparedInput {
                name: spec.name.clone(),
                shape,
                value: array.into_tensor(spec)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(PreparedInputs {
        inputs,
        source: InputSource::Archive,
    })
}

/// Generates every input from the model's declared shapes and dtypes.
fn synthesize_all(specs: &[InputSpec], options: SynthOptions) -> Result<PreparedInputs> {
    // One generator shared across every input, so a given --seed stays reproducible in input
    // order. Seeding per input would make every input of the same shape identical.
    let mut rng = StdRng::seed_from_u64(options.seed);

    let inputs = specs
        .iter()
        .map(|spec| {
            Ok(PreparedInput {
                name: spec.name.clone(),
                shape: spec.resolved_shape.clone(),
                value: synth::synthesize(spec, options.fill, &mut rng, options.int_max)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(PreparedInputs {
        inputs,
        source: InputSource::Synthesized,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::value::TensorElementType;

    fn spec(name: &str) -> InputSpec {
        InputSpec {
            name: name.into(),
            declared_shape: vec![1],
            symbolic_dims: vec![String::new()],
            resolved_shape: vec![1],
            element_type: TensorElementType::Float32,
        }
    }

    #[test]
    fn an_archive_supplying_exactly_the_inputs_matches() {
        let specs = [spec("input_a"), spec("input_b")];
        assert!(match_archive(&specs, &["input_a", "input_b"]).is_exact());
    }

    /// Order is irrelevant -- savez writes whatever order the kwargs came in.
    #[test]
    fn archive_order_does_not_matter() {
        let specs = [spec("input_a"), spec("input_b")];
        assert!(match_archive(&specs, &["input_b", "input_a"]).is_exact());
    }

    #[test]
    fn an_array_naming_no_input_is_a_mismatch() {
        let mismatch = match_archive(&[spec("input_a")], &["input_a", "typo_b"]);
        assert_eq!(mismatch.unused, vec!["typo_b".to_string()]);
        assert!(mismatch.missing.is_empty());
    }

    /// The case that motivates failing rather than warning: a half-supplied archive would
    /// otherwise be benchmarked half on real data and half on generated data.
    #[test]
    fn an_input_the_archive_omits_is_a_mismatch() {
        let mismatch = match_archive(&[spec("input_a"), spec("input_b")], &["input_a"]);
        assert_eq!(mismatch.missing, vec!["input_b".to_string()]);
        assert!(mismatch.unused.is_empty());
    }

    /// A typo shows up as both sides at once, and the message must name both.
    #[test]
    fn a_renamed_array_is_reported_as_both_missing_and_unused() {
        let mismatch = match_archive(&[spec("input_a")], &["input_A"]);
        assert_eq!(mismatch.missing, vec!["input_a".to_string()]);
        assert_eq!(mismatch.unused, vec!["input_A".to_string()]);

        let rendered = mismatch.to_string();
        assert!(rendered.contains("input_a"), "{rendered}");
        assert!(rendered.contains("input_A"), "{rendered}");
    }

    /// `HashMap` iteration order is arbitrary, so the message must not reorder between runs.
    #[test]
    fn mismatch_names_are_sorted() {
        let mismatch = match_archive(&[spec("a")], &["zeta", "alpha", "mid"]);
        assert_eq!(mismatch.unused, vec!["alpha", "mid", "zeta"]);
    }
}
