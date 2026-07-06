#pragma once

#include <onnxruntime_cxx_api.h>

#include "config.hpp"

namespace ort_runner {

// Owns the process-wide Ort::Env plus the loaded session. Constructing this does not itself
// measure anything; callers time the construction to get "model load time".
class OrtSessionHandle {
public:
    explicit OrtSessionHandle(const Config &config);

    Ort::Session &session() { return session_; }
    const Ort::Session &session() const { return session_; }

private:
    Ort::Env env_;
    Ort::SessionOptions options_;
    Ort::Session session_;
};

}  // namespace ort_runner
