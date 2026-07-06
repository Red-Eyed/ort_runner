#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "config.hpp"

namespace ort_runner {

struct InputSpec {
    std::string name;
    std::vector<int64_t> declared_shape;  // as reported by the model; dynamic dims are <= 0
    std::vector<int64_t> resolved_shape;  // after substituting default_dim for dynamic dims
    ONNXTensorElementDataType element_type;
};

// Pure function, no ORT session needed: substitutes any dim <= 0 (ORT's convention for both
// plain dynamic dims and named symbolic dims, both of which surface as -1 in GetShape()) with
// default_dim. Extracted so it can be unit-tested without a model or device.
std::vector<int64_t> ResolveShape(const std::vector<int64_t> &declared_shape, int64_t default_dim);

// Introspects the model's declared inputs. Throws std::runtime_error if an input has no shape
// information at all (fully unranked) -- auto-generation has no dimension count to work with
// in that case, and there's no override for it in v1.
std::vector<InputSpec> DescribeInputs(Ort::Session &session, int64_t default_dim);

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

struct SynthesizedInputs {
    std::vector<InputSpec> specs;
    std::vector<std::string> names;
    std::vector<std::vector<std::byte>> storage;  // owns each tensor's backing memory
    std::vector<Ort::Value> values;                // Ort::Value(s) wrapping `storage`, no copy
};

// Synthesizes one input tensor per model input, sized per its resolved shape and filled per
// `fill`. Throws std::runtime_error for an element type outside the supported subset
// (float32/float64/int64/int32/int16/int8/uint8/bool).
SynthesizedInputs SynthesizeInputs(Ort::Session &session, const Ort::MemoryInfo &memory_info,
                                    int64_t default_dim, FillStrategy fill, uint64_t seed,
                                    int64_t int_fill_max);

}  // namespace ort_runner
