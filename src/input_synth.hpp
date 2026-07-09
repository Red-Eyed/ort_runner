#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "config.hpp"

namespace ort_runner {

struct InputSpec {
    std::string name;
    std::vector<int64_t> declared_shape;     // as reported by the model; dynamic dims are <= 0
    std::vector<std::string> symbolic_dims;  // parallel to declared_shape; "" for a non-symbolic axis
    std::vector<int64_t> resolved_shape;     // after substituting default_dim/overrides for dynamic dims
    ONNXTensorElementDataType element_type;
};

// Pure function, no ORT session needed: for each axis, a static dim (> 0) passes through
// unchanged. A dynamic dim (<= 0) uses dim_overrides[symbolic_dims[i]] if that axis has a
// symbolic name present in the map, else default_dim. Extracted so it can be unit-tested
// without a model or device.
std::vector<int64_t> ResolveShape(const std::vector<int64_t> &declared_shape,
                                   const std::vector<std::string> &symbolic_dims,
                                   const DimOverrides &dim_overrides, int64_t default_dim);

// Introspects the model's declared inputs. Throws std::runtime_error if an input has no shape
// information at all (fully unranked) -- auto-generation has no dimension count to work with
// in that case, and there's no override for it in v1.
std::vector<InputSpec> DescribeInputs(Ort::Session &session, const DimOverrides &dim_overrides,
                                       int64_t default_dim);

// Prints a stderr warning (not a hard error) for any --dim override key that matched no
// symbolic dimension name across any input -- catches typos without being disruptive.
void WarnUnusedDimOverrides(const std::vector<InputSpec> &specs, const DimOverrides &dim_overrides);

std::string ElementTypeName(ONNXTensorElementDataType type);

struct OutputSpec {
    std::string name;
    std::vector<int64_t> shape;  // as declared; dynamic dims are <= 0, not substituted (ORT
                                  // allocates outputs itself; there is nothing to synthesize)
    ONNXTensorElementDataType element_type;
};

// Introspects the model's declared outputs -- their names are needed for Session::Run(), and
// the full spec backs --list-io.
std::vector<OutputSpec> DescribeOutputs(Ort::Session &session);

struct PreparedInputs {
    std::vector<InputSpec> specs;  // resolved_shape reflects the actual per-input shape used
    std::vector<std::string> names;
    std::vector<std::vector<std::byte>> storage;  // owns each tensor's backing memory
    std::vector<Ort::Value> values;                // Ort::Value(s) wrapping `storage`, no copy
    std::vector<std::string> sources;  // parallel to specs: "synth" or "file:<npz path>"
};

// Synthesizes one input tensor for `spec` into `storage`, returning an Ort::Value that views it.
// `rng` is threaded in (not seeded here) so a caller can share one generator across every
// synthesized input and keep fills seed-reproducible in input order. Throws std::runtime_error
// for an element type outside the supported subset
// (float32/float64/int64/int32/int16/int8/uint8/bool).
Ort::Value SynthesizeOneInput(std::vector<std::byte> &storage, const Ort::MemoryInfo &memory_info,
                              const InputSpec &spec, FillStrategy fill, std::mt19937_64 &rng,
                              int64_t int_fill_max);

}  // namespace ort_runner
