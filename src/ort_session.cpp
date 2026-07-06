#include "ort_session.hpp"

#ifdef __ANDROID__
#include <nnapi_provider_factory.h>
#else
#include <stdexcept>
#endif

namespace ort_runner {

namespace {

// ORT's own C API enums are plain, unscoped enums named identically to ours, so the
// unqualified names inside this file would resolve to our enum class instead -- return types
// are explicitly ::-qualified to disambiguate.
::ExecutionMode ToOrtExecutionMode(ExecutionMode mode) {
    return mode == ExecutionMode::parallel ? ORT_PARALLEL : ORT_SEQUENTIAL;
}

::GraphOptimizationLevel ToOrtGraphOptimizationLevel(GraphOptimizationLevel level) {
    switch (level) {
        case GraphOptimizationLevel::disable: return ORT_DISABLE_ALL;
        case GraphOptimizationLevel::basic: return ORT_ENABLE_BASIC;
        case GraphOptimizationLevel::extended: return ORT_ENABLE_EXTENDED;
        case GraphOptimizationLevel::layout: return ORT_ENABLE_LAYOUT;
        case GraphOptimizationLevel::all: return ORT_ENABLE_ALL;
    }
    return ORT_ENABLE_ALL;
}

Ort::SessionOptions BuildSessionOptions(const Config &config) {
    Ort::SessionOptions options;
    if (config.intra_op_threads.has_value()) {
        options.SetIntraOpNumThreads(*config.intra_op_threads);
    }
    options.SetInterOpNumThreads(config.inter_op_threads);
    options.SetExecutionMode(ToOrtExecutionMode(config.execution_mode));
    if (config.intra_op_spinning.has_value()) {
        options.AddConfigEntry("session.intra_op.allow_spinning",
                                *config.intra_op_spinning ? "1" : "0");
    }
    if (config.inter_op_spinning.has_value()) {
        options.AddConfigEntry("session.inter_op.allow_spinning",
                                *config.inter_op_spinning ? "1" : "0");
    }

    options.SetGraphOptimizationLevel(ToOrtGraphOptimizationLevel(config.graph_optimization_level));
    if (config.disable_cpu_arena) options.DisableCpuMemArena();
    if (config.disable_mem_pattern) options.DisableMemPattern();
    if (config.optimized_model_path.has_value()) {
        options.SetOptimizedModelFilePath(config.optimized_model_path->c_str());
    }
    options.SetLogSeverityLevel(static_cast<int>(config.log_severity));
    if (config.profile) {
        options.EnableProfiling(config.profile_prefix.c_str());
    }

    switch (config.provider) {
        case ExecutionProvider::cpu:
            break;  // implicit default, no registration call needed
        case ExecutionProvider::xnnpack:
            // Available in both the Linux and Android prebuilt packages (confirmed via
            // `strings` on both .so's: real xnnpack::* kernels are compiled in), registered
            // through ORT's newer generic string-based provider API rather than a dedicated
            // per-EP function -- no #ifdef __ANDROID__ guard needed, unlike NNAPI below.
            options.AppendExecutionProvider("XNNPACK");
            break;
        case ExecutionProvider::nnapi:
#ifdef __ANDROID__
            Ort::ThrowOnError(
                OrtSessionOptionsAppendExecutionProvider_Nnapi(options, /*nnapi_flags=*/0));
#else
            // config.cpp already rejects --provider nnapi on non-Android builds; this is
            // unreachable in practice and only guards against future callers bypassing that check.
            throw std::runtime_error("--provider nnapi is only supported in Android builds");
#endif
            break;
    }
    return options;
}

}  // namespace

OrtSessionHandle::OrtSessionHandle(const Config &config)
    : env_(static_cast<OrtLoggingLevel>(config.log_severity), "ort_runner"),
      options_(BuildSessionOptions(config)),
      session_(env_, config.model_path.c_str(), options_) {}

}  // namespace ort_runner
