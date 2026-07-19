//! Synthesizing input tensors from a model's declared shapes and dtypes.
//!
//! This is the feature the tool exists for: getting a latency number without hand-crafting
//! representative inputs. The values are meaningless numerically -- they only need to be
//! type-correct, shape-correct and cheap to produce.

use anyhow::Result;
use ort::value::DynValue;
use rand::rngs::StdRng;
use rand::Rng;

use crate::cli::Fill;
use crate::dispatch_dtype;
use crate::model::InputSpec;
use crate::tensors::dtype;

/// How to produce a vector of `count` values of one element type.
///
/// A trait rather than a free function because each strategy means something different per
/// type: "one" is `1.0` for a float and `true` for a bool, and "random" has no single sensible
/// range across floats, signed integers and booleans.
pub trait Fillable: Sized {
    fn zeros(count: usize) -> Vec<Self>;
    fn ones(count: usize) -> Vec<Self>;
    /// `int_max` is ignored by types for which it is meaningless (floats, bools).
    fn random(count: usize, rng: &mut StdRng, int_max: i64) -> Vec<Self>;
}

/// Floats fill from the unit interval, matching what the C++ used. The range is arbitrary but
/// deliberately small: activations far from zero make some models produce inf/NaN, which can
/// change how long an operator takes and so distort the measurement.
macro_rules! impl_fillable_float {
    ($($T:ty),*) => {$(
        impl Fillable for $T {
            fn zeros(count: usize) -> Vec<Self> { vec![0.0; count] }
            fn ones(count: usize) -> Vec<Self> { vec![1.0; count] }
            fn random(count: usize, rng: &mut StdRng, _int_max: i64) -> Vec<Self> {
                (0..count).map(|_| rng.random_range(0.0..1.0)).collect()
            }
        }
    )*};
}

/// Integers clamp to `int_max` (`--int-fill-max`, default 15).
///
/// Integer inputs are usually indices -- token ids, gather/embedding lookups -- and a value
/// beyond the table size makes ONNX Runtime abort the run with an index-out-of-range error.
/// Clamping low mitigates that. It is not a fix: a model whose vocabulary is smaller than the
/// clamp can still fail, and only the user knows the right bound.
macro_rules! impl_fillable_int {
    ($($T:ty),*) => {$(
        impl Fillable for $T {
            fn zeros(count: usize) -> Vec<Self> { vec![0; count] }
            fn ones(count: usize) -> Vec<Self> { vec![1; count] }
            fn random(count: usize, rng: &mut StdRng, int_max: i64) -> Vec<Self> {
                // Clamp to the type's own range first: --int-fill-max is a single i64 applied
                // to every integer input, so a default of 15 must not overflow an i8 tensor.
                //
                // Neither truncation nor sign loss is possible, precisely because of that
                // clamp: the value is already inside [0, $T::MAX] by the time it is narrowed.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let ceiling = int_max.clamp(0, i64::from(<$T>::MAX)) as $T;
                (0..count).map(|_| rng.random_range(0..=ceiling)).collect()
            }
        }
    )*};
}

impl_fillable_float!(f32, f64);
impl_fillable_int!(i64, i32, i16, i8, u8);

impl Fillable for bool {
    fn zeros(count: usize) -> Vec<Self> {
        vec![false; count]
    }
    fn ones(count: usize) -> Vec<Self> {
        vec![true; count]
    }
    fn random(count: usize, rng: &mut StdRng, _int_max: i64) -> Vec<Self> {
        (0..count).map(|_| rng.random()).collect()
    }
}

fn fill_values<T: Fillable>(count: usize, fill: Fill, rng: &mut StdRng, int_max: i64) -> Vec<T> {
    match fill {
        Fill::Zeros => T::zeros(count),
        Fill::Ones => T::ones(count),
        Fill::Random => T::random(count, rng, int_max),
    }
}

/// Synthesizes one input tensor for `spec`.
///
/// `rng` is threaded in rather than seeded here so one generator can be shared across every
/// input of a run: that keeps a given `--seed` reproducible in input order, which a per-input
/// generator would not.
///
/// # Errors
/// If the element type is unsupported, or the resolved shape is unusable.
pub fn synthesize(
    spec: &InputSpec,
    fill: Fill,
    rng: &mut StdRng,
    int_max: i64,
) -> Result<DynValue> {
    let count = dtype::element_count(&spec.resolved_shape)?;
    let shape = spec.resolved_shape.clone();

    dispatch_dtype!(spec.element_type, T => {
        let values = fill_values::<T>(count, fill, rng, int_max);
        Ok(ort::value::Tensor::from_array((shape, values))?.into_dyn())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn zeros_and_ones_are_exact() {
        assert_eq!(fill_values::<f32>(3, Fill::Zeros, &mut rng(), 15), vec![0.0, 0.0, 0.0]);
        assert_eq!(fill_values::<i32>(3, Fill::Ones, &mut rng(), 15), vec![1, 1, 1]);
        assert_eq!(fill_values::<bool>(2, Fill::Ones, &mut rng(), 15), vec![true, true]);
    }

    #[test]
    fn random_floats_stay_in_the_unit_interval() {
        let values = fill_values::<f32>(500, Fill::Random, &mut rng(), 15);
        assert!(values.iter().all(|v| (0.0..1.0).contains(v)), "out of range");
    }

    #[test]
    fn random_integers_respect_int_max() {
        let values = fill_values::<i32>(500, Fill::Random, &mut rng(), 7);
        assert!(values.iter().all(|v| (0..=7).contains(v)), "out of range: {values:?}");
    }

    /// --int-fill-max is one i64 applied to every integer input, so the default of 15 must not
    /// overflow a narrower tensor; i8 caps at 127 and u8 at 255.
    #[test]
    fn int_max_is_clamped_to_the_element_type() {
        let values = fill_values::<i8>(200, Fill::Random, &mut rng(), 100_000);
        assert!(values.iter().all(|v| *v >= 0), "should stay non-negative");

        let values = fill_values::<u8>(200, Fill::Random, &mut rng(), 100_000);
        assert!(!values.is_empty());
    }

    /// The same seed must produce the same tensors, or a benchmark is not reproducible.
    #[test]
    fn the_same_seed_reproduces_the_same_values() {
        let a = fill_values::<f64>(64, Fill::Random, &mut StdRng::seed_from_u64(7), 15);
        let b = fill_values::<f64>(64, Fill::Random, &mut StdRng::seed_from_u64(7), 15);
        assert_eq!(a, b);
    }

    /// One shared generator across inputs means the second tensor differs from the first;
    /// re-seeding per input would make every input identical.
    #[test]
    fn successive_draws_from_one_generator_differ() {
        let mut rng = rng();
        let first = fill_values::<f64>(32, Fill::Random, &mut rng, 15);
        let second = fill_values::<f64>(32, Fill::Random, &mut rng, 15);
        assert_ne!(first, second);
    }

    #[test]
    fn a_zero_element_request_yields_an_empty_vector() {
        assert!(fill_values::<f32>(0, Fill::Random, &mut rng(), 15).is_empty());
    }
}
