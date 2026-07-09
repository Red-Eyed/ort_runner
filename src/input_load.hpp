#pragma once

#include <onnxruntime_cxx_api.h>

#include "config.hpp"
#include "input_synth.hpp"

namespace ort_runner {

// Builds one input tensor per model input. When config.inputs_path is set, the named .npz is
// read once; for each model input whose name matches an array in the archive, that array's bytes
// back the tensor (validated: dtype and rank must match the model, and every statically-declared
// dimension must equal the file's -- dynamic dims take the file's size). Any input not present
// in the archive (or when no --inputs was given) is synthesized exactly as before. The returned
// specs' resolved_shape and the parallel `sources` vector reflect what each input actually got.
//
// Throws std::runtime_error on a validation failure, a missing/invalid archive, or an
// unsupported element type.
PreparedInputs BuildInputs(Ort::Session &session, const Ort::MemoryInfo &memory_info,
                           const Config &config);

}  // namespace ort_runner
