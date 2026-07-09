#include "input_synth.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace ort_runner {

namespace {

int64_t NumElements(const std::vector<int64_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
}

template <typename T>
void FillBuffer(T *data, int64_t count, FillStrategy fill, std::mt19937_64 &rng,
                int64_t int_fill_max) {
    if (fill == FillStrategy::zeros) {
        std::fill_n(data, count, static_cast<T>(0));
        return;
    }
    if (fill == FillStrategy::ones) {
        std::fill_n(data, count, static_cast<T>(1));
        return;
    }
    if constexpr (std::is_same_v<T, bool>) {
        std::uniform_int_distribution<int> dist(0, 1);
        std::generate_n(data, count, [&] { return dist(rng) != 0; });
    } else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(static_cast<T>(0), static_cast<T>(1));
        std::generate_n(data, count, [&] { return dist(rng); });
    } else {
        // std::uniform_int_distribution is only defined for short/int/long/long long (and
        // unsigned) by the standard; substitute `int` for the 1-byte integer types.
        using DistT = std::conditional_t<sizeof(T) == 1, int, T>;
        std::uniform_int_distribution<DistT> dist(0, static_cast<DistT>(int_fill_max));
        std::generate_n(data, count, [&] { return static_cast<T>(dist(rng)); });
    }
}

template <typename T>
Ort::Value MakeTensor(std::vector<std::byte> &storage, const Ort::MemoryInfo &memory_info,
                       const std::vector<int64_t> &shape, int64_t count, FillStrategy fill,
                       std::mt19937_64 &rng, int64_t int_fill_max) {
    storage.resize(static_cast<size_t>(count) * sizeof(T));
    auto *data = reinterpret_cast<T *>(storage.data());
    FillBuffer(data, count, fill, rng, int_fill_max);
    return Ort::Value::CreateTensor<T>(memory_info, data, static_cast<size_t>(count),
                                        shape.data(), shape.size());
}

}  // namespace

std::vector<int64_t> ResolveShape(const std::vector<int64_t> &declared_shape,
                                   const std::vector<std::string> &symbolic_dims,
                                   const DimOverrides &dim_overrides, int64_t default_dim) {
    std::vector<int64_t> resolved(declared_shape.size());
    for (size_t i = 0; i < declared_shape.size(); ++i) {
        if (declared_shape[i] > 0) {
            resolved[i] = declared_shape[i];
            continue;
        }
        const std::string &name = i < symbolic_dims.size() ? symbolic_dims[i] : std::string();
        auto override_it = name.empty() ? dim_overrides.end() : dim_overrides.find(name);
        resolved[i] = override_it != dim_overrides.end() ? override_it->second : default_dim;
    }
    return resolved;
}

std::string ElementTypeName(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
        default: return "unsupported(" + std::to_string(static_cast<int>(type)) + ")";
    }
}

std::vector<InputSpec> DescribeInputs(Ort::Session &session, const DimOverrides &dim_overrides,
                                       int64_t default_dim) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<InputSpec> specs;
    size_t count = session.GetInputCount();
    specs.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        auto name_ptr = session.GetInputNameAllocated(i, allocator);
        std::string name = name_ptr.get();

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        if (!tensor_info.HasShape()) {
            throw std::runtime_error("input '" + name +
                                      "' has no shape information at all (fully unranked); "
                                      "auto-generated inputs require at least a known rank");
        }

        InputSpec spec;
        spec.name = std::move(name);
        spec.declared_shape = tensor_info.GetShape();
        auto raw_symbolic_dims = tensor_info.GetSymbolicDimensions();
        spec.symbolic_dims.assign(raw_symbolic_dims.begin(), raw_symbolic_dims.end());
        spec.resolved_shape =
            ResolveShape(spec.declared_shape, spec.symbolic_dims, dim_overrides, default_dim);
        spec.element_type = tensor_info.GetElementType();
        specs.push_back(std::move(spec));
    }
    return specs;
}

void WarnUnusedDimOverrides(const std::vector<InputSpec> &specs,
                            const DimOverrides &dim_overrides) {
    std::vector<std::string> seen_names;
    for (const auto &spec : specs) {
        for (const auto &name : spec.symbolic_dims) {
            if (!name.empty()) seen_names.push_back(name);
        }
    }
    for (const auto &[name, value] : dim_overrides) {
        bool matched = std::find(seen_names.begin(), seen_names.end(), name) != seen_names.end();
        if (!matched) {
            std::cerr << "warning: --dim " << name << "=" << value
                      << " does not match any symbolic dimension name in this model's inputs\n";
        }
    }
}

std::vector<OutputSpec> DescribeOutputs(Ort::Session &session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<OutputSpec> specs;
    size_t count = session.GetOutputCount();
    specs.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        auto name_ptr = session.GetOutputNameAllocated(i, allocator);
        OutputSpec spec;
        spec.name = name_ptr.get();

        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        spec.shape = tensor_info.HasShape() ? tensor_info.GetShape() : std::vector<int64_t>{};
        spec.element_type = tensor_info.GetElementType();
        specs.push_back(std::move(spec));
    }
    return specs;
}

Ort::Value SynthesizeOneInput(std::vector<std::byte> &storage, const Ort::MemoryInfo &memory_info,
                              const InputSpec &spec, FillStrategy fill, std::mt19937_64 &rng,
                              int64_t int_fill_max) {
    int64_t count = NumElements(spec.resolved_shape);
    const auto &shape = spec.resolved_shape;

    switch (spec.element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return MakeTensor<float>(storage, memory_info, shape, count, fill, rng, int_fill_max);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return MakeTensor<double>(storage, memory_info, shape, count, fill, rng, int_fill_max);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return MakeTensor<int64_t>(storage, memory_info, shape, count, fill, rng, int_fill_max);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return MakeTensor<int32_t>(storage, memory_info, shape, count, fill, rng, int_fill_max);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return MakeTensor<int16_t>(storage, memory_info, shape, count, fill, rng, int_fill_max);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return MakeTensor<int8_t>(storage, memory_info, shape, count, fill, rng, int_fill_max);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return MakeTensor<uint8_t>(storage, memory_info, shape, count, fill, rng, int_fill_max);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return MakeTensor<bool>(storage, memory_info, shape, count, fill, rng, int_fill_max);
        default:
            throw std::runtime_error(
                "input '" + spec.name + "' has element type " + ElementTypeName(spec.element_type) +
                ", which is outside the subset auto-generation supports "
                "(float32/float64/int64/int32/int16/int8/uint8/bool)");
    }
}

}  // namespace ort_runner
