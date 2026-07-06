#include "report.hpp"

#include <onnxruntime_cxx_api.h>
#include <json.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>

namespace ort_runner {

namespace {

std::string ShapeToString(const std::vector<int64_t> &shape) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out << ", ";
        out << shape[i];
    }
    out << "]";
    return out.str();
}

std::string ThreadsToString(const Config &config) {
    std::ostringstream out;
    out << "intra=";
    if (config.intra_op_threads.has_value()) {
        out << *config.intra_op_threads;
    } else {
        out << "runtime default";
    }
    out << ", inter=" << config.inter_op_threads;
    return out.str();
}

}  // namespace

void PrintIoDescription(const std::vector<InputSpec> &inputs,
                         const std::vector<OutputSpec> &outputs) {
    std::cout << "inputs:\n";
    for (const auto &spec : inputs) {
        std::cout << "  " << spec.name << ": shape=" << ShapeToString(spec.declared_shape)
                   << " dtype=" << ElementTypeName(spec.element_type) << "\n";
    }
    std::cout << "outputs:\n";
    for (const auto &spec : outputs) {
        std::cout << "  " << spec.name << ": shape=" << ShapeToString(spec.shape)
                   << " dtype=" << ElementTypeName(spec.element_type) << "\n";
    }
}

void PrintPreamble(const Config &config, double load_time_ms,
                    const std::vector<InputSpec> &inputs,
                    const std::vector<OutputSpec> &outputs) {
    std::cout << "=== ort_runner ===\n";
    std::cout << "model:        " << config.model_path << "\n";
    std::cout << "ort_version:  " << Ort::GetVersionString() << "\n";
    std::cout << "provider:     " << ToString(config.provider) << "\n";
    std::cout << "threads:      " << ThreadsToString(config) << "\n";
    std::cout << "fill:         " << ToString(config.fill) << " (seed=" << config.seed << ")\n";
    std::cout << "load_time:    " << std::fixed << std::setprecision(3) << load_time_ms
               << " ms\n";
    std::cout << "\ninputs (post dynamic-dim substitution, default_dim=" << config.default_dim
               << "):\n";
    for (const auto &spec : inputs) {
        std::cout << "  " << spec.name << ": declared=" << ShapeToString(spec.declared_shape)
                   << " resolved=" << ShapeToString(spec.resolved_shape)
                   << " dtype=" << ElementTypeName(spec.element_type) << "\n";
    }
    std::cout << "outputs:\n";
    for (const auto &spec : outputs) {
        std::cout << "  " << spec.name << ": shape=" << ShapeToString(spec.shape)
                   << " dtype=" << ElementTypeName(spec.element_type) << "\n";
    }
    std::cout << "\n";
}

void PrintTrailer(long peak_rss_kb) {
    std::cout << "\npeak_rss:     " << peak_rss_kb << " KB (" << std::fixed
               << std::setprecision(1) << (static_cast<double>(peak_rss_kb) / 1024.0)
               << " MB)\n";
}

void PrintJsonReport(const Config &config, double load_time_ms,
                      const std::vector<InputSpec> &inputs,
                      const std::vector<OutputSpec> &outputs, const BenchmarkOutcome &outcome) {
    nlohmann::json inputs_json = nlohmann::json::array();
    for (const auto &spec : inputs) {
        inputs_json.push_back({
            {"name", spec.name},
            {"declared_shape", spec.declared_shape},
            {"resolved_shape", spec.resolved_shape},
            {"dtype", ElementTypeName(spec.element_type)},
        });
    }

    nlohmann::json outputs_json = nlohmann::json::array();
    for (const auto &spec : outputs) {
        outputs_json.push_back({
            {"name", spec.name},
            {"shape", spec.shape},
            {"dtype", ElementTypeName(spec.element_type)},
        });
    }

    nlohmann::json report;
    report["model_path"] = config.model_path;
    report["ort_version"] = Ort::GetVersionString();
    report["provider"] = ToString(config.provider);
    report["intra_op_threads"] = config.intra_op_threads.has_value()
                                      ? nlohmann::json(*config.intra_op_threads)
                                      : nlohmann::json("runtime default");
    report["inter_op_threads"] = config.inter_op_threads;
    report["fill"] = ToString(config.fill);
    report["seed"] = config.seed;
    report["default_dim"] = config.default_dim;
    report["load_time_ms"] = load_time_ms;
    report["peak_rss_kb"] = outcome.peak_rss_kb;
    report["inputs"] = inputs_json;
    report["outputs"] = outputs_json;
    report["benchmark"] = nlohmann::json::parse(outcome.nanobench_json);

    std::cout << report.dump(2) << "\n";
}

}  // namespace ort_runner
