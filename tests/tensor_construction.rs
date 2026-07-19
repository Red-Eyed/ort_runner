// Tests assert on known-good values, so unwrap/expect here is a deliberate "this must hold".
// lib.rs allows the same under cfg(test), but a tests/ file is its own crate and is not covered
// by it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The one part of the input-loading path that needs a real ONNX Runtime.
//!
//! `LoadedArray::into_tensor` validates against the model's declaration and then builds an
//! `ort::value::Tensor`. Validation is pure and unit-tested in `src/tensors/load.rs`; the build
//! step is not, because it initialises ONNX Runtime. The unit tests are required to pass with no
//! runtime present at all (see `src/lib.rs`), so the success path lives here instead.
//!
//! Ignored by default: `cargo test` must stay runnable on a machine that has never downloaded a
//! runtime. Run it with `just test-e2e`, which fetches the SDK and points this at it.

use std::path::PathBuf;

use ort::value::TensorElementType;
use ort_runner::model::InputSpec;
use ort_runner::tensors::load::parse_npy;

/// Where to load ONNX Runtime from. Set by `scripts/test_e2e.py`.
///
/// Deliberately not ORT_DYLIB_PATH: this binary calls `ort::init_from` explicitly, exactly as
/// `main.rs` does, so the variable names who sets it rather than implying ort reads it itself.
const DYLIB_ENV: &str = "ORT_RUNNER_TEST_DYLIB";

/// Loads ONNX Runtime, or fails with a message saying how to get one.
///
/// Checks the path before handing it to ort. Reaching ort without a usable library is the
/// failure this whole file exists to contain -- it does not return an error, it blocks forever
/// -- so a missing or mistyped path has to become a panic here, while it is still diagnosable.
fn init_ort() {
    let raw = std::env::var(DYLIB_ENV).unwrap_or_else(|_| {
        panic!("{DYLIB_ENV} is not set; run this via `just test-e2e`, not `cargo test --ignored`")
    });
    let path = PathBuf::from(&raw);
    assert!(path.is_file(), "{DYLIB_ENV}={raw} does not exist");

    ort::init_from(&path)
        .unwrap_or_else(|err| panic!("loading ONNX Runtime from {raw}: {err}"))
        .commit();
}

/// Builds a minimal little-endian float32 .npy buffer: magic, version, header length, header,
/// data. A trimmed copy of the unit tests' helper -- an integration test links against the
/// public library only, and this scaffolding is not part of it.
fn npy_f32(shape: &[usize], values: &[f32]) -> Vec<u8> {
    let shape_text = match shape {
        [single] => format!("({single},)"),
        dims => format!(
            "({})",
            dims.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}");
    while (10 + header.len() + 1) % 16 != 0 {
        header.push(' ');
    }
    header.push('\n');

    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    out
}

fn spec(name: &str, declared: Vec<i64>) -> InputSpec {
    InputSpec {
        name: name.into(),
        declared_shape: declared.clone(),
        symbolic_dims: vec![String::new(); declared.len()],
        resolved_shape: declared,
        element_type: TensorElementType::Float32,
    }
}

/// A dynamic dimension is defined by the data, so the file's size is accepted as-is.
#[test]
#[ignore = "needs a real ONNX Runtime; run via `just test-e2e`"]
fn a_dynamic_dimension_takes_the_file_size() {
    init_ort();

    let array = parse_npy(&npy_f32(&[3], &[1.0, 2.0, 3.0])).unwrap();
    let tensor = array.into_tensor(&spec("x", vec![-1])).unwrap();

    assert_eq!(tensor.shape().to_vec(), vec![3_i64]);
}

/// A statically declared dimension that agrees with the file also builds.
#[test]
#[ignore = "needs a real ONNX Runtime; run via `just test-e2e`"]
fn a_matching_static_shape_builds_a_tensor() {
    init_ort();

    let array = parse_npy(&npy_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0])).unwrap();
    let tensor = array.into_tensor(&spec("x", vec![2, 2])).unwrap();

    assert_eq!(tensor.shape().to_vec(), vec![2_i64, 2]);
}
