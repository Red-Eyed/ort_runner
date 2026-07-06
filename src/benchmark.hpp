#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

#include "config.hpp"
#include "input_synth.hpp"

namespace ort_runner {

struct BenchmarkOutcome {
    // Populated only when config.output_format == kJson: nanobench's own rendered JSON
    // (name/unit/median/mean/stddev/per-epoch samples), for report.cpp to merge with the
    // tool-specific fields nanobench has no way to know about. In human mode, nanobench
    // prints its own table directly to stdout during RunBenchmark() and this is left empty.
    std::string nanobench_json;
    long peak_rss_kb = 0;
};

BenchmarkOutcome RunBenchmark(Ort::Session &session, SynthesizedInputs &inputs,
                               const std::vector<std::string> &output_names,
                               const Config &config);

}  // namespace ort_runner
