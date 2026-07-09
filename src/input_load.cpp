#include "input_load.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "npz.hpp"

namespace ort_runner {

namespace {

ONNXTensorElementDataType ToOrtElementType(NpyDType dtype) {
    switch (dtype) {
        case NpyDType::f4: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case NpyDType::f8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case NpyDType::i8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case NpyDType::i4: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case NpyDType::i2: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case NpyDType::i1: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case NpyDType::u1: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case NpyDType::b1: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

std::string ShapeText(const std::vector<int64_t> &shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

// Validates a loaded array against the model's declared spec and, on success, sets the spec's
// resolved_shape to the file's actual shape (dynamic dims are defined by the data, not --dim).
void ValidateAndResolve(InputSpec &spec, const NpyArray &array) {
    ONNXTensorElementDataType file_type = ToOrtElementType(array.dtype);
    if (file_type != spec.element_type) {
        throw std::runtime_error("input '" + spec.name + "': .npz array dtype " +
                                 NpyDTypeName(array.dtype) + " does not match the model's dtype " +
                                 ElementTypeName(spec.element_type));
    }
    if (array.shape.size() != spec.declared_shape.size()) {
        throw std::runtime_error("input '" + spec.name + "': .npz array rank " +
                                 std::to_string(array.shape.size()) +
                                 " does not match the model's rank " +
                                 std::to_string(spec.declared_shape.size()));
    }
    for (size_t i = 0; i < spec.declared_shape.size(); ++i) {
        int64_t declared = spec.declared_shape[i];
        if (declared > 0 && declared != array.shape[i]) {
            throw std::runtime_error("input '" + spec.name + "': .npz array shape " +
                                     ShapeText(array.shape) +
                                     " conflicts with the model's declared shape " +
                                     ShapeText(spec.declared_shape) + " at axis " +
                                     std::to_string(i));
        }
    }
    spec.resolved_shape = array.shape;
}

// Copies the loaded bytes into `storage` and wraps them in an Ort::Value of the input's type.
Ort::Value MakeTensorFromLoaded(std::vector<std::byte> &storage,
                                const Ort::MemoryInfo &memory_info, const InputSpec &spec,
                                const NpyArray &array) {
    storage = array.data;  // storage must outlive the Ort::Value; PreparedInputs owns it
    return Ort::Value::CreateTensor(memory_info, storage.data(), storage.size(),
                                    spec.resolved_shape.data(), spec.resolved_shape.size(),
                                    spec.element_type);
}

// Warns (not fatal) about archive arrays that match no model input -- a likely name typo, but
// harmless, so it should not abort a run whose other inputs are valid.
void WarnUnusedArrays(const std::vector<InputSpec> &specs,
                      const std::unordered_map<std::string, NpyArray> &arrays) {
    for (const auto &[name, array] : arrays) {
        bool matched = std::any_of(specs.begin(), specs.end(),
                                   [&](const InputSpec &spec) { return spec.name == name; });
        if (!matched) {
            std::cerr << "warning: --inputs array '" << name
                      << "' matches no model input name and will be ignored\n";
        }
    }
}

}  // namespace

PreparedInputs BuildInputs(Ort::Session &session, const Ort::MemoryInfo &memory_info,
                           const Config &config) {
    PreparedInputs result;
    result.specs = DescribeInputs(session, config.dim_overrides, config.default_dim);

    std::unordered_map<std::string, NpyArray> loaded;
    if (config.inputs_path.has_value()) {
        loaded = LoadNpz(*config.inputs_path);
        WarnUnusedArrays(result.specs, loaded);
    }

    result.names.reserve(result.specs.size());
    result.storage.reserve(result.specs.size());
    result.values.reserve(result.specs.size());
    result.sources.reserve(result.specs.size());

    std::mt19937_64 rng(config.seed);

    for (auto &spec : result.specs) {
        result.names.push_back(spec.name);
        result.storage.emplace_back();
        auto &buf = result.storage.back();

        auto it = loaded.find(spec.name);
        if (it != loaded.end()) {
            ValidateAndResolve(spec, it->second);
            result.values.push_back(MakeTensorFromLoaded(buf, memory_info, spec, it->second));
            result.sources.push_back("file:" + *config.inputs_path);
        } else {
            result.values.push_back(
                SynthesizeOneInput(buf, memory_info, spec, config.fill, rng, config.int_fill_max));
            result.sources.push_back("synth");
        }
    }

    return result;
}

}  // namespace ort_runner
