// Tests assert on known-good values, so unwrap/expect there is a deliberate "this must hold"
// rather than the unhandled-error smell the lint targets in library code.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

//! `ort_runner`: benchmark ONNX Runtime inference with auto-generated input tensors.
//!
//! Split into a library so integration tests can drive the real APIs. The modules below the
//! `ort` boundary (`cli`, and later the shape/dtype/stats logic) are pure and testable with no
//! ONNX Runtime present at all -- which is what lets `cargo test` run on a host that has never
//! downloaded a runtime.

pub mod cli;
pub mod dylib;
pub mod info;
pub mod model;
pub mod session;
