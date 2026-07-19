//! Bridging a *runtime* tensor element type to *compile-time* Rust types.
//!
//! Tensor element types are only known when the model is read, but `Tensor<T>` is generic, so
//! every operation on tensor data has to cross that boundary. The C++ crossed it with a
//! function template instantiated from a switch; the equivalent here is a macro that expands
//! one match arm per supported type, so the crossing is written once rather than repeated in
//! every function that needs it.

use anyhow::{bail, Result};
use ort::value::TensorElementType;

/// The element types ort_runner can synthesize or load.
///
/// Matches the C++ subset. Everything outside it (float16, bfloat16, string, complex, the
/// 4-bit and 8-bit float types) is rejected with a clear message rather than silently
/// mishandled: they need either a non-trivial fill strategy or a non-primitive layout.
pub const SUPPORTED: &[TensorElementType] = &[
    TensorElementType::Float32,
    TensorElementType::Float64,
    TensorElementType::Int64,
    TensorElementType::Int32,
    TensorElementType::Int16,
    TensorElementType::Int8,
    TensorElementType::Uint8,
    TensorElementType::Bool,
];

/// Runs `$body` with `$T` bound to the Rust type matching a runtime [`TensorElementType`].
///
/// Written as a macro rather than a generic function because the whole point is to *choose* a
/// type, which a generic function cannot do -- its caller would have to know the type already.
///
/// ```ignore
/// let tensor = dispatch_dtype!(spec.element_type, T => make_tensor::<T>(&shape)?);
/// ```
#[macro_export]
macro_rules! dispatch_dtype {
    ($element_type:expr, $T:ident => $body:expr) => {{
        use ort::value::TensorElementType as Ty;
        match $element_type {
            Ty::Float32 => { type $T = f32; $body }
            Ty::Float64 => { type $T = f64; $body }
            Ty::Int64 => { type $T = i64; $body }
            Ty::Int32 => { type $T = i32; $body }
            Ty::Int16 => { type $T = i16; $body }
            Ty::Int8 => { type $T = i8; $body }
            Ty::Uint8 => { type $T = u8; $body }
            Ty::Bool => { type $T = bool; $body }
            other => {
                return Err($crate::tensors::dtype::unsupported(other));
            }
        }
    }};
}

/// The error returned for an element type outside [`SUPPORTED`].
///
/// Shared so every dispatch site reports the same thing, including the list of what *is*
/// supported -- otherwise the user has to guess which types would work.
#[must_use]
pub fn unsupported(element_type: TensorElementType) -> anyhow::Error {
    let supported: Vec<String> =
        SUPPORTED.iter().map(|ty| crate::model::element_type_name(*ty)).collect();
    anyhow::anyhow!(
        "element type {} is outside the subset ort_runner supports ({})",
        crate::model::element_type_name(element_type),
        supported.join(", ")
    )
}

/// Rejects an unsupported element type up front.
///
/// Called before any allocation so a model with an unusable input fails immediately, rather
/// than after synthesizing several megabytes of other inputs.
///
/// # Errors
/// If `element_type` is outside [`SUPPORTED`].
pub fn ensure_supported(element_type: TensorElementType) -> Result<()> {
    if !SUPPORTED.contains(&element_type) {
        bail!(unsupported(element_type));
    }
    Ok(())
}

/// Number of elements in a shape.
///
/// Returns 1 for an empty shape, which is correct: a rank-0 shape describes a scalar tensor
/// holding exactly one element, not an empty one.
///
/// # Errors
/// If any dimension is negative (an unresolved dynamic dimension reaching this point is a bug,
/// not user error) or the product overflows.
pub fn element_count(shape: &[i64]) -> Result<usize> {
    let mut total: i64 = 1;
    for &dim in shape {
        if dim < 0 {
            bail!("shape {shape:?} still contains an unresolved dynamic dimension");
        }
        total = total
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("shape {shape:?} overflows when multiplied out"))?;
    }
    usize::try_from(total).map_err(|_| anyhow::anyhow!("shape {shape:?} does not fit in usize"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_a_normal_shape() {
        assert_eq!(element_count(&[2, 3, 4]).unwrap(), 24);
    }

    /// A rank-0 tensor holds one scalar, so the empty product is 1 rather than 0.
    #[test]
    fn an_empty_shape_is_one_element() {
        assert_eq!(element_count(&[]).unwrap(), 1);
    }

    /// A legitimately zero-sized dimension yields zero elements, which is not an error.
    #[test]
    fn a_zero_dimension_yields_no_elements() {
        assert_eq!(element_count(&[2, 0, 4]).unwrap(), 0);
    }

    #[test]
    fn rejects_an_unresolved_dynamic_dimension() {
        let err = element_count(&[-1, 3]).unwrap_err().to_string();
        assert!(err.contains("unresolved dynamic dimension"), "{err}");
    }

    #[test]
    fn rejects_an_overflowing_shape() {
        assert!(element_count(&[i64::MAX, 2]).is_err());
    }

    #[test]
    fn accepts_every_supported_type() {
        for ty in SUPPORTED {
            assert!(ensure_supported(*ty).is_ok(), "{ty:?} should be supported");
        }
    }

    #[test]
    fn rejects_an_unsupported_type_and_lists_alternatives() {
        let err = ensure_supported(TensorElementType::Float16).unwrap_err().to_string();
        assert!(err.contains("float16"), "{err}");
        assert!(err.contains("float32"), "should list what is supported: {err}");
    }
}
