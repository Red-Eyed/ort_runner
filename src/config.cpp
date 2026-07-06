#include "config.hpp"

#include <argparse.hpp>
#include <onnxruntime_cxx_api.h>

#include <iostream>

#include "enum_utils.hpp"

namespace ort_runner {

namespace {

std::string VersionString() {
    return "ort_runner 0.1.0 (onnxruntime " + Ort::GetVersionString() + ")";
}

// Parses repeated "--dim name=value" entries into a DimOverrides map. Returns false (and
// prints a usage error) on a malformed entry, matching the same "invalid args" failure path
// as argparse's own validation.
bool ParseDimOverrides(const std::vector<std::string> &entries, argparse::ArgumentParser &program,
                       DimOverrides &out) {
    for (const auto &entry : entries) {
        auto eq = entry.find('=');
        if (eq == std::string::npos || eq == 0 || eq == entry.size() - 1) {
            std::cerr << "invalid --dim value '" << entry << "' (expected name=value)\n\n"
                      << program;
            return false;
        }
        std::string name = entry.substr(0, eq);
        std::string value_str = entry.substr(eq + 1);
        try {
            size_t consumed = 0;
            long long value = std::stoll(value_str, &consumed);
            if (consumed != value_str.size()) throw std::invalid_argument(value_str);
            out[name] = value;
        } catch (const std::exception &) {
            std::cerr << "invalid --dim value '" << entry << "' (value must be an integer)\n\n"
                      << program;
            return false;
        }
    }
    return true;
}

}  // namespace

std::optional<Config> ParseArgs(int argc, const char *const argv[]) {
    argparse::ArgumentParser program("ort_runner", VersionString());
    program.add_description(
        "Loads an ONNX model, auto-generates inputs from its declared shapes/dtypes, and "
        "benchmarks inference latency and peak memory usage.");

    program.add_argument("--model").required().help("path to the .onnx model file");

    program.add_argument("--warmup")
        .default_value(uint64_t{3})
        .scan<'u', uint64_t>()
        .help("untimed iterations before measurement (forwarded to nanobench's warmup())");

    program.add_argument("--min-epoch-iterations")
        .default_value(uint64_t{1})
        .scan<'u', uint64_t>()
        .help("minimum iterations per measurement epoch (forwarded to nanobench's "
              "minEpochIterations())");

    program.add_argument("--threads")
        .scan<'i', int>()
        .help("intra-op thread count; left at onnxruntime's own default if not passed");

    program.add_argument("--inter-op-threads")
        .default_value(1)
        .scan<'i', int>()
        .help("inter-op thread count");

    program.add_argument("--execution-mode")
        .default_value(std::string{"sequential"})
        .choices("sequential", "parallel")
        .help("ORT graph execution mode; parallel only helps multi-branch graphs with "
              "inter-op-threads > 1");

    program.add_argument("--intra-op-spinning")
        .choices("on", "off")
        .help("allow intra-op threads to spin instead of sleep; left at onnxruntime's own "
              "default if not passed");

    program.add_argument("--inter-op-spinning")
        .choices("on", "off")
        .help("allow inter-op threads to spin instead of sleep; left at onnxruntime's own "
              "default if not passed");

    program.add_argument("--provider")
        .default_value(std::string{"cpu"})
        .choices("cpu", "nnapi", "xnnpack")
        .help("execution provider; 'nnapi' is only valid in Android builds");

    program.add_argument("--default-dim")
        .default_value(int64_t{1})
        .scan<'i', int64_t>()
        .help("value substituted for any dynamic/symbolic input dimension not covered by --dim");

    program.add_argument("--dim")
        .append()
        .default_value(std::vector<std::string>{})
        .help("override a named symbolic input dimension, e.g. --dim batch=4 (repeatable; "
              "falls back to --default-dim for unmatched or anonymous dynamic dims)");

    program.add_argument("--fill")
        .default_value(std::string{"random"})
        .choices("random", "ones", "zeros")
        .help("input tensor fill strategy");

    program.add_argument("--seed")
        .default_value(uint64_t{42})
        .scan<'u', uint64_t>()
        .help("RNG seed used when --fill=random");

    program.add_argument("--int-fill-max")
        .default_value(int64_t{15})
        .scan<'i', int64_t>()
        .help("clamp for randomly-filled integer tensors (mitigates embedding/gather "
              "index-out-of-range crashes; a mitigation, not a full fix)");

    program.add_argument("--output-format")
        .default_value(std::string{"human"})
        .choices("human", "json")
        .help("human-readable text, or a single merged JSON document");

    program.add_argument("--list-io")
        .flag()
        .help("print declared input/output names, shapes, and dtypes, then exit without "
              "running inference");

    program.add_argument("--profile")
        .flag()
        .help("enable onnxruntime's built-in per-op profiler (writes a Chrome-trace-format "
              "JSON file; the path is printed after the run)");

    program.add_argument("--profile-prefix")
        .default_value(std::string{"ort_runner_profile"})
        .help("file prefix passed to onnxruntime's profiler when --profile is set");

    program.add_argument("--graph-optimization-level")
        .default_value(std::string{"all"})
        .choices("disable", "basic", "extended", "layout", "all")
        .help("onnxruntime graph optimization level");

    program.add_argument("--disable-cpu-arena")
        .flag()
        .help("disable the CPU memory arena (relevant for accurate peak-RSS measurement, "
              "since the arena pre-allocates)");

    program.add_argument("--disable-mem-pattern")
        .flag()
        .help("disable memory pattern optimization");

    program.add_argument("--optimized-model-path")
        .help("dump the post-optimization graph to this path; unset means don't dump");

    program.add_argument("--log-severity")
        .default_value(std::string{"warning"})
        .choices("verbose", "info", "warning", "error", "fatal")
        .help("onnxruntime session log severity level");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << "\n\n" << program;
        return std::nullopt;
    }

    Config config;
    config.model_path = program.get<std::string>("--model");
    config.warmup_iterations = program.get<uint64_t>("--warmup");
    config.min_epoch_iterations = program.get<uint64_t>("--min-epoch-iterations");
    config.intra_op_threads = program.present<int>("--threads");
    config.inter_op_threads = program.get<int>("--inter-op-threads");
    config.execution_mode = FromString<ExecutionMode>(program.get<std::string>("--execution-mode"));
    if (auto v = program.present<std::string>("--intra-op-spinning")) {
        config.intra_op_spinning = (*v == "on");
    }
    if (auto v = program.present<std::string>("--inter-op-spinning")) {
        config.inter_op_spinning = (*v == "on");
    }
    config.provider = FromString<ExecutionProvider>(program.get<std::string>("--provider"));
    config.default_dim = program.get<int64_t>("--default-dim");
    if (!ParseDimOverrides(program.get<std::vector<std::string>>("--dim"), program,
                           config.dim_overrides)) {
        return std::nullopt;
    }
    config.fill = FromString<FillStrategy>(program.get<std::string>("--fill"));
    config.seed = program.get<uint64_t>("--seed");
    config.int_fill_max = program.get<int64_t>("--int-fill-max");
    config.output_format = FromString<OutputFormat>(program.get<std::string>("--output-format"));
    config.list_io = program.get<bool>("--list-io");
    config.profile = program.get<bool>("--profile");
    config.profile_prefix = program.get<std::string>("--profile-prefix");
    config.graph_optimization_level =
        FromString<GraphOptimizationLevel>(program.get<std::string>("--graph-optimization-level"));
    config.disable_cpu_arena = program.get<bool>("--disable-cpu-arena");
    config.disable_mem_pattern = program.get<bool>("--disable-mem-pattern");
    config.optimized_model_path = program.present<std::string>("--optimized-model-path");
    config.log_severity = FromString<LogSeverity>(program.get<std::string>("--log-severity"));

#ifndef __ANDROID__
    if (config.provider == ExecutionProvider::nnapi) {
        std::cerr << "--provider nnapi is only supported in Android builds\n";
        return std::nullopt;
    }
#endif

    return config;
}

}  // namespace ort_runner
