#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

#include "config.hpp"
#include "input_synth.hpp"

namespace ort_runner {

// Mirrors pandas' df.describe(): summary statistics over the per-epoch latency samples
// nanobench recorded (one sample per measured epoch, in milliseconds).
struct DescribeStats {
    uint64_t count = 0;
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double p25 = 0.0;
    double median = 0.0;
    double p75 = 0.0;
    double max = 0.0;
};

struct BenchmarkStats {
    uint64_t warmup_iterations = 0;    // skipped, unmeasured (config.warmup_iterations)
    uint64_t measured_epochs = 0;      // nanobench's measurement unit; one latency sample each
    uint64_t total_iterations = 0;     // total session.Run() calls across all measured epochs
    double total_measured_time_s = 0.0;
    double throughput_ops_per_sec = 0.0;
    DescribeStats latency_ms;
};

struct BenchmarkOutcome {
    // Populated only when config.output_format == kJson: nanobench's own rendered JSON
    // (name/unit/median/mean/stddev/per-epoch samples), for report.cpp to merge with the
    // tool-specific fields nanobench has no way to know about.
    std::string nanobench_json;
    BenchmarkStats stats;
    long peak_rss_kb = 0;
};

BenchmarkOutcome RunBenchmark(Ort::Session &session, SynthesizedInputs &inputs,
                               const std::vector<std::string> &output_names,
                               const Config &config);

}  // namespace ort_runner
