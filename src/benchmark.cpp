#include "benchmark.hpp"

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>

namespace ort_runner {

namespace {

long PeakRssKb() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

// Linear-interpolation quantile, matching numpy/pandas' default method.
double Quantile(const std::vector<double> &sorted_values, double q) {
    if (sorted_values.size() == 1) return sorted_values.front();
    double pos = q * static_cast<double>(sorted_values.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return sorted_values[lo];
    double frac = pos - static_cast<double>(lo);
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}

DescribeStats ComputeDescribeStats(std::vector<double> values) {
    std::sort(values.begin(), values.end());

    DescribeStats stats;
    stats.count = values.size();
    stats.min = values.front();
    stats.max = values.back();
    stats.p25 = Quantile(values, 0.25);
    stats.median = Quantile(values, 0.50);
    stats.p75 = Quantile(values, 0.75);

    double sum = 0.0;
    for (double v : values) sum += v;
    stats.mean = sum / static_cast<double>(values.size());

    if (values.size() > 1) {
        double squared_diff_sum = 0.0;
        for (double v : values) {
            double diff = v - stats.mean;
            squared_diff_sum += diff * diff;
        }
        // Sample standard deviation (ddof=1), matching pandas' describe() default.
        stats.stddev = std::sqrt(squared_diff_sum / static_cast<double>(values.size() - 1));
    }
    return stats;
}

// One sample per measured epoch: nanobench stores each epoch's average per-iteration elapsed
// time (in seconds), which is the finest-grained latency data it exposes.
std::vector<double> EpochLatenciesMs(const ankerl::nanobench::Result &result) {
    std::vector<double> latencies_ms;
    latencies_ms.reserve(result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        latencies_ms.push_back(result.get(i, ankerl::nanobench::Result::Measure::elapsed) *
                                1000.0);
    }
    return latencies_ms;
}

BenchmarkStats ComputeBenchmarkStats(const ankerl::nanobench::Result &result,
                                      uint64_t warmup_iterations) {
    using Measure = ankerl::nanobench::Result::Measure;

    BenchmarkStats stats;
    stats.warmup_iterations = warmup_iterations;
    stats.measured_epochs = result.size();
    stats.total_iterations = static_cast<uint64_t>(result.sum(Measure::iterations));
    stats.total_measured_time_s = result.sumProduct(Measure::iterations, Measure::elapsed);
    stats.latency_ms = ComputeDescribeStats(EpochLatenciesMs(result));
    stats.throughput_ops_per_sec =
        stats.latency_ms.mean > 0.0 ? 1000.0 / stats.latency_ms.mean : 0.0;
    return stats;
}

}  // namespace

BenchmarkOutcome RunBenchmark(Ort::Session &session, SynthesizedInputs &inputs,
                               const std::vector<std::string> &output_names,
                               const Config &config) {
    std::vector<const char *> input_name_ptrs;
    input_name_ptrs.reserve(inputs.names.size());
    for (const auto &name : inputs.names) input_name_ptrs.push_back(name.c_str());

    std::vector<const char *> output_name_ptrs;
    output_name_ptrs.reserve(output_names.size());
    for (const auto &name : output_names) output_name_ptrs.push_back(name.c_str());

    // Constructed once, outside the timed lambda, so CreateRunOptions() doesn't pollute
    // per-iteration latency.
    Ort::RunOptions run_options;

    ankerl::nanobench::Bench bench;
    bench.title("ort_runner inference");
    bench.warmup(config.warmup_iterations);
    bench.minEpochIterations(config.min_epoch_iterations);
    // Suppress nanobench's own console table in every mode; report.cpp renders our own
    // summary (human) or merges the JSON template (json) instead.
    bench.output(nullptr);

    bench.run("inference", [&] {
        auto outputs = session.Run(run_options, input_name_ptrs.data(), inputs.values.data(),
                                    inputs.values.size(), output_name_ptrs.data(),
                                    output_name_ptrs.size());
        ankerl::nanobench::doNotOptimizeAway(outputs);
    });

    BenchmarkOutcome outcome;
    outcome.stats = ComputeBenchmarkStats(bench.results().front(), config.warmup_iterations);
    if (config.output_format == OutputFormat::json) {
        std::ostringstream json_capture;
        bench.render(ankerl::nanobench::templates::json(), json_capture);
        outcome.nanobench_json = json_capture.str();
    }
    outcome.peak_rss_kb = PeakRssKb();
    return outcome;
}

}  // namespace ort_runner
