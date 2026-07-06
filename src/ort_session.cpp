#include "ort_session.hpp"

#ifdef __ANDROID__
#include <nnapi_provider_factory.h>
#else
#include <stdexcept>
#endif

namespace ort_runner {

namespace {

Ort::SessionOptions BuildSessionOptions(const Config &config) {
    Ort::SessionOptions options;
    if (config.intra_op_threads.has_value()) {
        options.SetIntraOpNumThreads(*config.intra_op_threads);
    }
    options.SetInterOpNumThreads(config.inter_op_threads);

    if (config.provider == ExecutionProvider::kNnapi) {
#ifdef __ANDROID__
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(options, /*nnapi_flags=*/0));
#else
        // config.cpp already rejects --provider nnapi on non-Android builds; this is
        // unreachable in practice and only guards against future callers bypassing that check.
        throw std::runtime_error("--provider nnapi is only supported in Android builds");
#endif
    }
    return options;
}

}  // namespace

OrtSessionHandle::OrtSessionHandle(const Config &config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ort_runner"),
      options_(BuildSessionOptions(config)),
      session_(env_, config.model_path.c_str(), options_) {}

}  // namespace ort_runner
