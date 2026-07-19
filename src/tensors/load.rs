//! Loading real input tensors from a `.npz` archive (`numpy.savez`).
//!
//! Synthetic inputs are enough for a latency number, but not for anything that depends on the
//! values -- a model whose control flow or sparsity varies with the data, or a sanity check
//! that the outputs are right. `--inputs` covers that case.
//!
//! A `.npz` is a zip of `.npy` members, so `zip` walks the archive and `npyz` parses each
//! member. Both are driven directly rather than through npyz's `npz` feature, which pulls
//! zstd and bzip2 (see Cargo.toml).

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context, Result};
use npyz::{Order, TypeChar};
use ort::value::{DynValue, TensorElementType};

use crate::dispatch_dtype;
use crate::model::{element_type_name, InputSpec};
use crate::tensors::dtype;

/// One array read from the archive, held as raw bytes until validated against a model input.
///
/// The bytes are kept undecoded so validation can reject a mismatch before any per-element
/// work happens: a wrong dtype or rank is cheap to detect and expensive to discover halfway
/// through converting a large tensor.
pub struct LoadedArray {
    pub element_type: TensorElementType,
    pub shape: Vec<i64>,
    raw: Vec<u8>,
}

impl std::fmt::Debug for LoadedArray {
    /// Deliberately omits the payload: these arrays are routinely megabytes, and a test
    /// failure or error report wants the shape, not the contents.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedArray")
            .field("element_type", &element_type_name(self.element_type))
            .field("shape", &self.shape)
            .field("bytes", &self.raw.len())
            .finish()
    }
}

/// Converting a typed slice to and from the raw bytes `LoadedArray` holds.
///
/// Exists for `bool` alone. Every other supported element type is `Pod`, so bytemuck can
/// reinterpret it in place, but Rust's `bool` has invalid bit patterns (only 0 and 1 are legal)
/// and bytemuck correctly refuses to transmute arbitrary bytes into one. numpy stores each
/// boolean as a single 0/1 byte, so the conversion is trivial -- it just cannot be a cast.
trait ByteRepr: Sized {
    fn to_bytes(values: &[Self]) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Vec<Self>;
}

/// Implemented per type rather than as a blanket `impl<T: Pod>`: a blanket impl would make a
/// separate `impl for bool` a coherence conflict, since nothing tells the compiler that `bool`
/// will never be `Pod`.
macro_rules! impl_byte_repr_pod {
    ($($T:ty),*) => {$(
        impl ByteRepr for $T {
            fn to_bytes(values: &[Self]) -> Vec<u8> {
                bytemuck::cast_slice::<Self, u8>(values).to_vec()
            }
            fn from_bytes(bytes: &[u8]) -> Vec<Self> {
                bytemuck::cast_slice::<u8, Self>(bytes).to_vec()
            }
        }
    )*};
}

impl_byte_repr_pod!(f32, f64, i64, i32, i16, i8, u8);

impl ByteRepr for bool {
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        values.iter().map(|&v| u8::from(v)).collect()
    }
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        // numpy writes 0 or 1, but treating anything non-zero as true avoids constructing an
        // invalid bool from a malformed file -- which would be undefined behaviour under a cast.
        bytes.iter().map(|&b| b != 0).collect()
    }
}

/// Reads every array in a `.npz`, keyed by the name `numpy.savez` was called with.
///
/// Member names carry a `.npy` suffix that the caller never sees, so it is stripped here --
/// the key then matches a model input name directly.
///
/// # Errors
/// If the file cannot be read, is not a valid zip, or holds an array this tool cannot use
/// (Fortran order, an unsupported dtype, or a pickled object array).
pub fn read_npz(path: &Path) -> Result<HashMap<String, LoadedArray>> {
    let file =
        File::open(path).with_context(|| format!("opening --inputs archive {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("reading {} as a zip archive (.npz)", path.display()))?;

    let mut arrays = HashMap::new();
    for index in 0..archive.len() {
        let mut member = archive.by_index(index)?;
        // strip_suffix, not trim_end_matches: the latter strips every repetition, so an array
        // legitimately named "data.npy" would arrive as "data.npy.npy" and come back as "data".
        let name = member.name();
        let name = name.strip_suffix(".npy").unwrap_or(name).to_string();

        let mut bytes = Vec::with_capacity(usize::try_from(member.size()).unwrap_or(0));
        member.read_to_end(&mut bytes)?;

        let array = parse_npy(&bytes)
            .with_context(|| format!("parsing array '{name}' from {}", path.display()))?;
        arrays.insert(name, array);
    }
    Ok(arrays)
}

/// Parses one in-memory `.npy` buffer.
///
/// Separate from the archive walk so it is testable against a byte buffer, with no file and no
/// zip involved.
///
/// # Errors
/// If the header is malformed, the array is Fortran-ordered, or the dtype is unsupported.
pub fn parse_npy(bytes: &[u8]) -> Result<LoadedArray> {
    let npy = npyz::NpyFile::new(std::io::Cursor::new(bytes))?;

    if npy.order() == Order::Fortran {
        bail!(
            "array is Fortran-ordered (column-major); ONNX Runtime tensors are row-major. \
             Re-save it with numpy.ascontiguousarray()"
        );
    }
    if npy.uses_pickled_array() {
        bail!("array is a pickled object array, which holds no plain numeric data");
    }

    let element_type = element_type_of(&npy.dtype())?;
    let shape: Vec<i64> = npy
        .shape()
        .iter()
        .map(|&d| i64::try_from(d).unwrap_or(i64::MAX))
        .collect();

    // into_vec handles any byte-order conversion, so a big-endian archive is read correctly
    // rather than rejected -- the C++ refused these outright.
    let raw = dispatch_dtype!(element_type, T => {
        let values: Vec<T> = npy.into_vec::<T>()?;
        <T as ByteRepr>::to_bytes(&values)
    });

    Ok(LoadedArray {
        element_type,
        shape,
        raw,
    })
}

/// Maps a numpy dtype onto an ONNX Runtime element type.
fn element_type_of(dtype: &npyz::DType) -> Result<TensorElementType> {
    let npyz::DType::Plain(type_str) = dtype else {
        bail!("structured or nested dtypes are not supported; expected a plain numeric array");
    };
    let size = type_str.num_bytes().unwrap_or(0);

    let element_type = match (type_str.type_char(), size) {
        (TypeChar::Float, 4) => TensorElementType::Float32,
        (TypeChar::Float, 8) => TensorElementType::Float64,
        (TypeChar::Int, 8) => TensorElementType::Int64,
        (TypeChar::Int, 4) => TensorElementType::Int32,
        (TypeChar::Int, 2) => TensorElementType::Int16,
        (TypeChar::Int, 1) => TensorElementType::Int8,
        (TypeChar::Uint, 1) => TensorElementType::Uint8,
        (TypeChar::Bool, _) => TensorElementType::Bool,
        (other, bytes) => bail!(
            "numpy dtype '{}{}' is not one ort_runner supports",
            other.to_str(),
            bytes
        ),
    };
    dtype::ensure_supported(element_type)?;
    Ok(element_type)
}

impl LoadedArray {
    /// Checks this array against the model's declaration and turns it into a tensor.
    ///
    /// A dynamic dimension takes the file's size -- the data defines it, so `--dim` and
    /// `--default-dim` do not apply to a loaded input. A statically declared dimension must
    /// match exactly.
    ///
    /// # Errors
    /// If dtype, rank, or any static dimension disagrees with the model.
    pub fn into_tensor(self, spec: &InputSpec) -> Result<DynValue> {
        if self.element_type != spec.element_type {
            bail!(
                "input '{}': archive dtype {} does not match the model's {}",
                spec.name,
                element_type_name(self.element_type),
                element_type_name(spec.element_type)
            );
        }
        if self.shape.len() != spec.declared_shape.len() {
            bail!(
                "input '{}': archive rank {} does not match the model's rank {}",
                spec.name,
                self.shape.len(),
                spec.declared_shape.len()
            );
        }
        for (axis, (&declared, &actual)) in spec
            .declared_shape
            .iter()
            .zip(self.shape.iter())
            .enumerate()
        {
            if declared > 0 && declared != actual {
                bail!(
                    "input '{}': archive shape {:?} conflicts with the model's declared shape \
                     {:?} at axis {axis}",
                    spec.name,
                    self.shape,
                    spec.declared_shape
                );
            }
        }

        let shape = self.shape;
        let raw = self.raw;
        dispatch_dtype!(self.element_type, T => {
            let values: Vec<T> = <T as ByteRepr>::from_bytes(&raw);
            Ok(ort::value::Tensor::from_array((shape, values))?.into_dyn())
        })
    }

    /// The shape the file actually holds, which becomes the input's resolved shape.
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal .npy v1.0 buffer: magic, version, header length, header, data.
    fn npy_bytes(descr: &str, fortran: bool, shape: &[usize], data: &[u8]) -> Vec<u8> {
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
        let mut header = format!(
            "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape_text}, }}",
            if fortran { "True" } else { "False" }
        );
        // The header must be padded so data starts on a 16-byte boundary.
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(' ');
        }
        header.push('\n');

        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn parses_a_float32_array() {
        let data: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let array = parse_npy(&npy_bytes("<f4", false, &[2, 2], &data)).unwrap();

        assert_eq!(array.element_type, TensorElementType::Float32);
        assert_eq!(array.shape, vec![2, 2]);
    }

    #[test]
    fn parses_an_int64_array() {
        let data: Vec<u8> = [7_i64, 8, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
        let array = parse_npy(&npy_bytes("<i8", false, &[3], &data)).unwrap();

        assert_eq!(array.element_type, TensorElementType::Int64);
        assert_eq!(array.shape, vec![3]);
    }

    /// numpy writes big-endian when asked; `into_vec` converts, so this is read rather than
    /// rejected the way the C++ rejected it.
    #[test]
    fn reads_a_big_endian_array() {
        let data: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let array = parse_npy(&npy_bytes(">f4", false, &[2], &data)).unwrap();

        assert_eq!(array.element_type, TensorElementType::Float32);
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&array.raw), &[1.0, 2.0]);
    }

    #[test]
    fn rejects_fortran_order() {
        let data: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let err = parse_npy(&npy_bytes("<f4", true, &[2], &data))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Fortran"), "{err}");
    }

    #[test]
    fn rejects_an_unsupported_dtype() {
        // float16 is a valid numpy dtype but outside the supported subset.
        let err = parse_npy(&npy_bytes("<f2", false, &[2], &[0, 0, 0, 0]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("not one ort_runner supports") || err.contains("outside the subset"),
            "{err}"
        );
    }

    fn spec(name: &str, declared: Vec<i64>, ty: TensorElementType) -> InputSpec {
        InputSpec {
            name: name.into(),
            declared_shape: declared.clone(),
            symbolic_dims: vec![String::new(); declared.len()],
            resolved_shape: declared,
            element_type: ty,
        }
    }

    #[test]
    fn rejects_a_dtype_mismatch_against_the_model() {
        let data: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let array = parse_npy(&npy_bytes("<f4", false, &[2], &data)).unwrap();

        let err = array
            .into_tensor(&spec("x", vec![2], TensorElementType::Int64))
            .unwrap_err()
            .to_string();
        assert!(err.contains("float32") && err.contains("int64"), "{err}");
    }

    #[test]
    fn rejects_a_rank_mismatch() {
        let data: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let array = parse_npy(&npy_bytes("<f4", false, &[2], &data)).unwrap();

        let err = array
            .into_tensor(&spec("x", vec![1, 2], TensorElementType::Float32))
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank"), "{err}");
    }

    #[test]
    fn rejects_a_static_dimension_mismatch() {
        let data: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let array = parse_npy(&npy_bytes("<f4", false, &[2], &data)).unwrap();

        let err = array
            .into_tensor(&spec("x", vec![5], TensorElementType::Float32))
            .unwrap_err()
            .to_string();
        assert!(err.contains("axis 0"), "{err}");
    }

    // The success path -- validation passing and a tensor actually being built -- is not tested
    // here. Constructing an ort Tensor initialises ONNX Runtime, and these tests are required to
    // run with no runtime present (see lib.rs). A test that reached it did not fail; it blocked
    // forever in ORT's lazy init, hanging the whole suite. It lives in
    // tests/tensor_construction.rs, which loads a real library first.
}
