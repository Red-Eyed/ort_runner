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
                                    const DimOverrides &dim_overrides, int64_t default_dim,
                                    FillStrategy fill, uint64_t seed, int64_t int_fill_max);

}  // namespace ort_runner
