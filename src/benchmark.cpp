#include "benchmark.hpp"

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <sys/resource.h>

#include <sstream>

namespace ort_runner {

namespace {

long PeakRssKb() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
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

    bool json_output = config.output_format == OutputFormat::kJson;
    if (json_output) {
        // Suppress nanobench's automatic console table; we render its JSON template instead
        // and hand it to report.cpp to merge with this tool's own fields.
        bench.output(nullptr);
    }

    bench.run("inference", [&] {
        auto outputs = session.Run(run_options, input_name_ptrs.data(), inputs.values.data(),
                                    inputs.values.size(), output_name_ptrs.data(),
                                    output_name_ptrs.size());
        ankerl::nanobench::doNotOptimizeAway(outputs);
    });

    BenchmarkOutcome outcome;
    if (json_output) {
        std::ostringstream json_capture;
        bench.render(ankerl::nanobench::templates::json(), json_capture);
        outcome.nanobench_json = json_capture.str();
    }
    outcome.peak_rss_kb = PeakRssKb();
    return outcome;
}

}  // namespace ort_runner
