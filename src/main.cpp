#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "benchmark.hpp"
#include "config.hpp"
#include "input_synth.hpp"
#include "ort_session.hpp"
#include "report.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

int main(int argc, char **argv) {
    auto config = ort_runner::ParseArgs(argc, argv);
    if (!config) {
        return 1;
    }

    try {
        auto load_start = Clock::now();
        ort_runner::OrtSessionHandle session_handle(*config);
        double load_time_ms = ElapsedMs(load_start, Clock::now());

        auto &session = session_handle.session();
        auto input_specs =
            ort_runner::DescribeInputs(session, config->dim_overrides, config->default_dim);
        ort_runner::WarnUnusedDimOverrides(input_specs, config->dim_overrides);
        auto output_specs = ort_runner::DescribeOutputs(session);

        if (config->list_io) {
            ort_runner::PrintIoDescription(input_specs, output_specs);
            return 0;
        }

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputs = ort_runner::SynthesizeInputs(session, memory_info, config->dim_overrides,
                                                    config->default_dim, config->fill,
                                                    config->seed, config->int_fill_max);

        std::vector<std::string> output_names;
        output_names.reserve(output_specs.size());
        for (const auto &spec : output_specs) output_names.push_back(spec.name);

        if (config->output_format == ort_runner::OutputFormat::human) {
            ort_runner::PrintPreamble(*config, load_time_ms, input_specs, output_specs);
        }

        auto outcome = ort_runner::RunBenchmark(session, inputs, output_names, *config);

        std::optional<std::string> profile_file;
        if (config->profile) {
            Ort::AllocatorWithDefaultOptions allocator;
            auto profile_path = session.EndProfilingAllocated(allocator);
            profile_file = std::string(profile_path.get());
        }

        if (config->output_format == ort_runner::OutputFormat::human) {
            ort_runner::PrintBenchmarkSummary(outcome.stats);
            ort_runner::PrintTrailer(outcome.peak_rss_kb, profile_file);
        } else {
            ort_runner::PrintJsonReport(*config, load_time_ms, input_specs, output_specs,
                                        outcome, profile_file);
        }
    } catch (const Ort::Exception &err) {
        std::cerr << "onnxruntime error: " << err.what() << "\n";
        return 1;
    } catch (const std::exception &err) {
        std::cerr << "error: " << err.what() << "\n";
        return 1;
    }

    return 0;
}
