// Tests assert on known-good values, so unwrap/expect there is a deliberate "this must hold"
// rather than the unhandled-error smell the lint targets in library code.
//
// float_cmp likewise: the statistics tests assert on values that pass through verbatim -- a
// nearest-rank percentile returns one of its inputs unchanged, and min/max are copies -- so exact
// equality is the precise claim. Where a value is genuinely computed (a mean, a standard
// deviation) those tests compare against an epsilon instead.
#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)
)]

//! `ort_runner`: benchmark ONNX Runtime inference with auto-generated input tensors.
//!
//! Split into a library so integration tests can drive the real APIs. The modules below the
//! `ort` boundary (`cli`, and later the shape/dtype/stats logic) are pure and testable with no
//! ONNX Runtime present at all -- which is what lets `cargo test` run on a host that has never
//! downloaded a runtime.

pub mod bench;
pub mod cli;
pub mod config;
pub mod dylib;
pub mod host;
pub mod info;
pub mod model;
pub mod paths;
pub mod profile;
pub mod progress;
pub mod qnn;
pub mod report;
pub mod session;
pub mod shutdown;
pub mod stats;
pub mod tensors;
pub mod watchdog;
