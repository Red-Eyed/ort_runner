//! Model introspection: what the graph declares, and what shape we will actually feed it.
//!
//! `resolve_shape` is deliberately free of any ONNX Runtime types so the dimension-substitution
//! rules -- the part with real branching -- are unit-testable without a model, a runtime, or a
//! device.

use anyhow::{bail, Result};
use ort::session::Session;
use ort::value::{TensorElementType, ValueType};
use serde::Serialize;

use crate::cli::DimOverrides;

#[derive(Debug, Clone, Serialize)]
pub struct InputSpec {
    pub name: String,
    /// As reported by the model; dynamic dimensions are negative.
    pub declared_shape: Vec<i64>,
    /// Parallel to `declared_shape`; empty string for an axis the graph did not name.
    pub symbolic_dims: Vec<String>,
    /// After substituting `--dim` / `--default-dim` for dynamic dimensions.
    pub resolved_shape: Vec<i64>,
    #[serde(serialize_with = "serialize_element_type")]
    pub element_type: TensorElementType,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputSpec {
    pub name: String,
    /// As declared. Dynamic dimensions are left negative rather than substituted: ONNX Runtime
    /// allocates outputs itself, so there is nothing here for us to choose.
    pub shape: Vec<i64>,
    #[serde(serialize_with = "serialize_element_type")]
    pub element_type: TensorElementType,
}

/// Substitutes a concrete size for every dynamic dimension.
///
/// A static dimension (positive) passes through untouched. A dynamic one (zero or negative)
/// takes `dim_overrides[name]` when that axis carries a symbolic name present in the map, and
/// `default_dim` otherwise -- so an anonymous dynamic axis is only ever reachable via
/// `--default-dim`.
#[must_use]
pub fn resolve_shape(
    declared_shape: &[i64],
    symbolic_dims: &[String],
    dim_overrides: &DimOverrides,
    default_dim: i64,
) -> Vec<i64> {
    declared_shape
        .iter()
        .enumerate()
        .map(|(axis, &declared)| {
            if declared > 0 {
                return declared;
            }
            symbolic_dims
                .get(axis)
                .filter(|name| !name.is_empty())
                .and_then(|name| dim_overrides.get(name))
                .copied()
                .unwrap_or(default_dim)
        })
        .collect()
}

/// numpy-style dtype name, matching what the C++ printed (`float32`, not `Float32`).
#[must_use]
pub fn element_type_name(ty: TensorElementType) -> String {
    match ty {
        TensorElementType::Float32 => "float32".into(),
        TensorElementType::Float64 => "float64".into(),
        TensorElementType::Float16 => "float16".into(),
        TensorElementType::Bfloat16 => "bfloat16".into(),
        TensorElementType::Int64 => "int64".into(),
        TensorElementType::Int32 => "int32".into(),
        TensorElementType::Int16 => "int16".into(),
        TensorElementType::Int8 => "int8".into(),
        TensorElementType::Uint64 => "uint64".into(),
        TensorElementType::Uint32 => "uint32".into(),
        TensorElementType::Uint16 => "uint16".into(),
        TensorElementType::Uint8 => "uint8".into(),
        TensorElementType::Bool => "bool".into(),
        TensorElementType::String => "string".into(),
        other => format!("unsupported({other:?})"),
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn serialize_element_type<S: serde::Serializer>(
    ty: &TensorElementType,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&element_type_name(*ty))
}

/// Describes the model's declared inputs, resolving each dynamic dimension.
///
/// # Errors
/// If an input is not a tensor, or is fully unranked -- auto-generation has no dimension count
/// to work from in either case, and there is no flag to supply one.
pub fn describe_inputs(
    session: &Session,
    dim_overrides: &DimOverrides,
    default_dim: i64,
) -> Result<Vec<InputSpec>> {
    session
        .inputs()
        .iter()
        .map(|outlet| {
            let ValueType::Tensor {
                ty,
                shape,
                dimension_symbols,
            } = outlet.dtype()
            else {
                bail!(
                    "input '{}' is not a tensor; auto-generated inputs only support tensors",
                    outlet.name()
                );
            };

            let declared_shape: Vec<i64> = shape.iter().copied().collect();
            let symbolic_dims: Vec<String> =
                dimension_symbols.iter().map(ToString::to_string).collect();
            let resolved_shape =
                resolve_shape(&declared_shape, &symbolic_dims, dim_overrides, default_dim);

            Ok(InputSpec {
                name: outlet.name().to_string(),
                declared_shape,
                symbolic_dims,
                resolved_shape,
                element_type: *ty,
            })
        })
        .collect()
}

/// Describes the model's declared outputs. Their names are needed to call `run()`; the rest
/// backs `--list-io`.
///
/// # Errors
/// If an output is not a tensor.
pub fn describe_outputs(session: &Session) -> Result<Vec<OutputSpec>> {
    session
        .outputs()
        .iter()
        .map(|outlet| {
            let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
                bail!("output '{}' is not a tensor", outlet.name());
            };
            Ok(OutputSpec {
                name: outlet.name().to_string(),
                shape: shape.iter().copied().collect(),
                element_type: *ty,
            })
        })
        .collect()
}

/// `--dim` names that matched no symbolic dimension anywhere in the model.
///
/// Returned rather than printed so the caller decides the channel and severity; a typo here is
/// worth a warning but must not abort a run whose other dimensions are fine.
#[must_use]
pub fn unmatched_dim_overrides<'a>(
    specs: &[InputSpec],
    dim_overrides: &'a DimOverrides,
) -> Vec<(&'a String, &'a i64)> {
    dim_overrides
        .iter()
        .filter(|(name, _)| {
            !specs
                .iter()
                .any(|spec| spec.symbolic_dims.iter().any(|dim| dim == *name))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overrides(pairs: &[(&str, i64)]) -> DimOverrides {
        pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
    }

    fn symbols(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn static_dims_are_left_untouched() {
        let declared = [1, 3, 224, 224];
        assert_eq!(resolve_shape(&declared, &[], &overrides(&[]), 8), declared);
    }

    #[test]
    fn negative_dims_are_substituted_with_default_dim() {
        assert_eq!(
            resolve_shape(&[-1, 3, 224, 224], &[], &overrides(&[]), 8),
            vec![8, 3, 224, 224]
        );
    }

    #[test]
    fn zero_valued_dims_are_treated_as_dynamic_too() {
        assert_eq!(resolve_shape(&[0, 3], &[], &overrides(&[]), 5), vec![5, 3]);
    }

    #[test]
    fn a_fully_dynamic_shape_resolves_every_dim() {
        assert_eq!(
            resolve_shape(&[-1, -1, -1], &[], &overrides(&[]), 2),
            vec![2, 2, 2]
        );
    }

    #[test]
    fn an_empty_shape_resolves_to_empty() {
        assert!(resolve_shape(&[], &[], &overrides(&[]), 4).is_empty());
    }

    #[test]
    fn a_named_symbolic_dim_uses_its_matching_override() {
        assert_eq!(
            resolve_shape(
                &[-1, 3, 224, 224],
                &symbols(&["batch", "", "", ""]),
                &overrides(&[("batch", 16)]),
                8
            ),
            vec![16, 3, 224, 224]
        );
    }

    #[test]
    fn an_override_naming_a_different_axis_is_ignored() {
        assert_eq!(
            resolve_shape(
                &[-1, 3],
                &symbols(&["batch", ""]),
                &overrides(&[("seq_len", 99)]),
                8
            ),
            vec![8, 3]
        );
    }

    #[test]
    fn an_anonymous_dynamic_dim_falls_back_to_default_dim() {
        assert_eq!(
            resolve_shape(&[-1], &symbols(&[""]), &overrides(&[("batch", 16)]), 8),
            vec![8]
        );
    }

    #[test]
    fn two_axes_sharing_a_symbolic_name_both_get_the_override() {
        assert_eq!(
            resolve_shape(&[-1, -1], &symbols(&["N", "N"]), &overrides(&[("N", 5)]), 1),
            vec![5, 5]
        );
    }

    /// A symbolic name list shorter than the shape must not panic -- the two come from
    /// separate ONNX Runtime accessors, so nothing structurally guarantees equal length.
    #[test]
    fn a_short_symbolic_dims_list_falls_back_instead_of_panicking() {
        assert_eq!(
            resolve_shape(&[-1, -1], &symbols(&["N"]), &overrides(&[("N", 5)]), 7),
            vec![5, 7]
        );
    }

    #[test]
    fn unmatched_overrides_are_reported() {
        let specs = vec![InputSpec {
            name: "x".into(),
            declared_shape: vec![-1],
            symbolic_dims: symbols(&["batch"]),
            resolved_shape: vec![1],
            element_type: TensorElementType::Float32,
        }];
        let overrides = overrides(&[("batch", 2), ("typo", 3)]);

        let unmatched = unmatched_dim_overrides(&specs, &overrides);

        assert_eq!(unmatched.len(), 1);
        assert_eq!(unmatched[0].0, "typo");
    }
}
