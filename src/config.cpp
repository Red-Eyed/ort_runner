#include "config.hpp"

#include <argparse.hpp>
#include <onnxruntime_cxx_api.h>

#include <iostream>

namespace ort_runner {

namespace {

ExecutionProvider ParseProvider(const std::string &value) {
    if (value == "cpu") return ExecutionProvider::kCpu;
    return ExecutionProvider::kNnapi;  // argparse's choices() already validated the value
}

FillStrategy ParseFill(const std::string &value) {
    if (value == "ones") return FillStrategy::kOnes;
    if (value == "zeros") return FillStrategy::kZeros;
    return FillStrategy::kRandom;
}

OutputFormat ParseOutputFormat(const std::string &value) {
    return value == "json" ? OutputFormat::kJson : OutputFormat::kHuman;
}

std::string VersionString() {
    return "ort_runner 0.1.0 (onnxruntime " + Ort::GetVersionString() + ")";
}

}  // namespace

std::string ToString(ExecutionProvider provider) {
    switch (provider) {
        case ExecutionProvider::kCpu: return "cpu";
        case ExecutionProvider::kNnapi: return "nnapi";
    }
    return "unknown";
}

std::string ToString(FillStrategy fill) {
    switch (fill) {
        case FillStrategy::kRandom: return "random";
        case FillStrategy::kOnes: return "ones";
        case FillStrategy::kZeros: return "zeros";
    }
    return "unknown";
}

std::string ToString(OutputFormat format) {
    switch (format) {
        case OutputFormat::kHuman: return "human";
        case OutputFormat::kJson: return "json";
    }
    return "unknown";
}

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

    program.add_argument("--provider")
        .default_value(std::string{"cpu"})
        .choices("cpu", "nnapi")
        .help("execution provider; 'nnapi' is only valid in Android builds");

    program.add_argument("--default-dim")
        .default_value(int64_t{1})
        .scan<'i', int64_t>()
        .help("value substituted for any dynamic/symbolic input dimension");

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
    config.provider = ParseProvider(program.get<std::string>("--provider"));
    config.default_dim = program.get<int64_t>("--default-dim");
    config.fill = ParseFill(program.get<std::string>("--fill"));
    config.seed = program.get<uint64_t>("--seed");
    config.int_fill_max = program.get<int64_t>("--int-fill-max");
    config.output_format = ParseOutputFormat(program.get<std::string>("--output-format"));
    config.list_io = program.get<bool>("--list-io");

#ifndef __ANDROID__
    if (config.provider == ExecutionProvider::kNnapi) {
        std::cerr << "--provider nnapi is only supported in Android builds\n";
        return std::nullopt;
    }
#endif

    return config;
}

}  // namespace ort_runner
