#include "report.hpp"

#include <onnxruntime_cxx_api.h>
#include <json.hpp>
#include <sys/utsname.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include "enum_utils.hpp"

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

std::string SymbolicDimsToString(const std::vector<std::string> &symbolic_dims) {
    std::vector<std::string> named;
    for (const auto &name : symbolic_dims) {
        if (!name.empty()) named.push_back(name);
    }
    if (named.empty()) return "(none)";
    std::ostringstream out;
    for (size_t i = 0; i < named.size(); ++i) {
        if (i > 0) out << ", ";
        out << named[i];
    }
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

std::string SpinningToString(const std::optional<bool> &spinning) {
    if (!spinning.has_value()) return "runtime default";
    return *spinning ? "on" : "off";
}

std::string ArenaPatternToString(const Config &config) {
    std::ostringstream out;
    out << "cpu_arena=" << (config.disable_cpu_arena ? "disabled" : "enabled")
        << ", mem_pattern=" << (config.disable_mem_pattern ? "disabled" : "enabled");
    return out.str();
}

std::string CurrentUtcTimestamp() {
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc_tm{};
    gmtime_r(&now, &utc_tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
    return std::string(buf);
}

struct HostInfo {
    std::string os_name;
    std::string os_release;
    std::string arch;
    unsigned int cpu_cores;
};

HostInfo CollectHostInfo() {
    struct utsname info {};
    uname(&info);
    return HostInfo{info.sysname, info.release, info.machine,
                     std::thread::hardware_concurrency()};
}

std::string FormatHostInfo(const HostInfo &host) {
    std::ostringstream out;
    out << host.os_name << " " << host.os_release << " (" << host.arch << "), " << host.cpu_cores
        << " cores";
    return out.str();
}

// The model was already loaded successfully by the time any report is printed, so the file is
// known to exist; std::error_code just avoids an exception for the display-only size lookup.
std::optional<uintmax_t> ModelFileSizeBytes(const std::string &model_path) {
    std::error_code ec;
    auto size = std::filesystem::file_size(model_path, ec);
    if (ec) return std::nullopt;
    return size;
}

}  // namespace

void PrintIoDescription(const std::vector<InputSpec> &inputs,
                         const std::vector<OutputSpec> &outputs) {
    std::cout << "inputs:\n";
    for (const auto &spec : inputs) {
        std::cout << "  " << spec.name << ": shape=" << ShapeToString(spec.declared_shape)
                   << " dtype=" << ElementTypeName(spec.element_type)
                   << " symbolic_dims=" << SymbolicDimsToString(spec.symbolic_dims) << "\n";
    }
    std::cout << "outputs:\n";
    for (const auto &spec : outputs) {
        std::cout << "  " << spec.name << ": shape=" << ShapeToString(spec.shape)
                   << " dtype=" << ElementTypeName(spec.element_type) << "\n";
    }
}

void PrintPreamble(const Config &config, double load_time_ms,
                    const std::vector<InputSpec> &inputs,
                    const std::vector<std::string> &input_sources,
                    const std::vector<OutputSpec> &outputs) {
    std::cout << "=== ort_runner ===\n";
    std::cout << "timestamp:          " << CurrentUtcTimestamp() << "\n";
    std::cout << "host:               " << FormatHostInfo(CollectHostInfo()) << "\n";
    std::cout << "model:              " << config.model_path;
    if (auto size_bytes = ModelFileSizeBytes(config.model_path)) {
        std::cout << " (" << std::fixed << std::setprecision(1)
                   << (static_cast<double>(*size_bytes) / (1024.0 * 1024.0)) << " MB)";
    }
    std::cout << "\n";
    std::cout << "ort_version:        " << Ort::GetVersionString() << "\n";
    std::cout << "provider:           " << ToString(config.provider) << "\n";
    std::cout << "threads:            " << ThreadsToString(config) << "\n";
    std::cout << "execution_mode:     " << ToString(config.execution_mode) << "\n";
    std::cout << "spinning:           intra=" << SpinningToString(config.intra_op_spinning)
               << ", inter=" << SpinningToString(config.inter_op_spinning) << "\n";
    std::cout << "graph_opt_level:    " << ToString(config.graph_optimization_level) << "\n";
    std::cout << "memory:             " << ArenaPatternToString(config) << "\n";
    std::cout << "log_severity:       " << ToString(config.log_severity) << "\n";
    if (config.optimized_model_path.has_value()) {
        std::cout << "optimized_model:    " << *config.optimized_model_path << "\n";
    }
    std::cout << "fill:               " << ToString(config.fill) << " (seed=" << config.seed
               << ")\n";
    std::cout << "load_time:          " << std::fixed << std::setprecision(3) << load_time_ms
               << " ms\n";
    std::cout << "\ninputs (post dynamic-dim substitution, default_dim=" << config.default_dim
               << "):\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &spec = inputs[i];
        std::cout << "  " << spec.name << ": declared=" << ShapeToString(spec.declared_shape)
                   << " resolved=" << ShapeToString(spec.resolved_shape)
                   << " dtype=" << ElementTypeName(spec.element_type)
                   << " symbolic_dims=" << SymbolicDimsToString(spec.symbolic_dims)
                   << " source=" << input_sources[i] << "\n";
    }
    std::cout << "outputs:\n";
    for (const auto &spec : outputs) {
        std::cout << "  " << spec.name << ": shape=" << ShapeToString(spec.shape)
                   << " dtype=" << ElementTypeName(spec.element_type) << "\n";
    }
    std::cout << "\n";
}

void PrintBenchmarkSummary(const BenchmarkStats &stats) {
    std::cout << "\nbenchmark: inference\n";
    std::cout << "  warmup_runs (skipped, unmeasured):  " << stats.warmup_iterations << "\n";
    std::cout << "  measured_runs (epochs):             " << stats.measured_epochs << "\n";
    std::cout << "  total_iterations:                   " << stats.total_iterations << "\n";
    std::cout << "  total_measured_time:                " << std::fixed
               << std::setprecision(3) << stats.total_measured_time_s << " s\n";
    std::cout << "  throughput:                         " << std::setprecision(2)
               << stats.throughput_ops_per_sec << " inferences/sec\n";

    std::cout << "\n  latency_ms:\n";
    auto print_stat = [](const char *label, double value) {
        std::cout << "    " << std::left << std::setw(8) << label << std::right << std::fixed
                   << std::setprecision(3) << std::setw(12) << value << "\n";
    };
    const auto &d = stats.latency_ms;
    print_stat("count", static_cast<double>(d.count));
    print_stat("mean", d.mean);
    print_stat("std", d.stddev);
    print_stat("min", d.min);
    print_stat("25%", d.p25);
    print_stat("50%", d.median);
    print_stat("75%", d.p75);
    print_stat("max", d.max);
}

void PrintTrailer(long peak_rss_kb, const std::optional<std::string> &profile_file) {
    std::cout << "\npeak_rss:     " << peak_rss_kb << " KB (" << std::fixed
               << std::setprecision(1) << (static_cast<double>(peak_rss_kb) / 1024.0)
               << " MB)\n";
    if (profile_file.has_value()) {
        std::cout << "profile:      " << *profile_file << "\n";
    }
}

void PrintJsonReport(const Config &config, double load_time_ms,
                      const std::vector<InputSpec> &inputs,
                      const std::vector<std::string> &input_sources,
                      const std::vector<OutputSpec> &outputs, const BenchmarkOutcome &outcome,
                      const std::optional<std::string> &profile_file) {
    nlohmann::json inputs_json = nlohmann::json::array();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &spec = inputs[i];
        inputs_json.push_back({
            {"name", spec.name},
            {"declared_shape", spec.declared_shape},
            {"symbolic_dims", spec.symbolic_dims},
            {"resolved_shape", spec.resolved_shape},
            {"dtype", ElementTypeName(spec.element_type)},
            {"source", input_sources[i]},
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

    nlohmann::json dim_overrides_json = nlohmann::json::object();
    for (const auto &[name, value] : config.dim_overrides) {
        dim_overrides_json[name] = value;
    }

    HostInfo host = CollectHostInfo();

    nlohmann::json report;
    report["timestamp"] = CurrentUtcTimestamp();
    report["host"] = {
        {"os", host.os_name + " " + host.os_release},
        {"arch", host.arch},
        {"cpu_cores", host.cpu_cores},
    };
    report["model_path"] = config.model_path;
    report["inputs_path"] = config.inputs_path.has_value() ? nlohmann::json(*config.inputs_path)
                                                           : nlohmann::json(nullptr);
    auto model_size_bytes = ModelFileSizeBytes(config.model_path);
    report["model_size_bytes"] =
        model_size_bytes.has_value() ? nlohmann::json(*model_size_bytes) : nlohmann::json(nullptr);
    report["ort_version"] = Ort::GetVersionString();
    report["provider"] = ToString(config.provider);
    report["intra_op_threads"] = config.intra_op_threads.has_value()
                                      ? nlohmann::json(*config.intra_op_threads)
                                      : nlohmann::json("runtime default");
    report["inter_op_threads"] = config.inter_op_threads;
    report["execution_mode"] = ToString(config.execution_mode);
    report["intra_op_spinning"] = SpinningToString(config.intra_op_spinning);
    report["inter_op_spinning"] = SpinningToString(config.inter_op_spinning);
    report["graph_optimization_level"] = ToString(config.graph_optimization_level);
    report["disable_cpu_arena"] = config.disable_cpu_arena;
    report["disable_mem_pattern"] = config.disable_mem_pattern;
    report["optimized_model_path"] = config.optimized_model_path.has_value()
                                          ? nlohmann::json(*config.optimized_model_path)
                                          : nlohmann::json(nullptr);
    report["log_severity"] = ToString(config.log_severity);
    report["fill"] = ToString(config.fill);
    report["seed"] = config.seed;
    report["default_dim"] = config.default_dim;
    report["dim_overrides"] = dim_overrides_json;
    report["load_time_ms"] = load_time_ms;
    report["peak_rss_kb"] = outcome.peak_rss_kb;
    report["profile_file"] =
        profile_file.has_value() ? nlohmann::json(*profile_file) : nlohmann::json(nullptr);
    report["inputs"] = inputs_json;
    report["outputs"] = outputs_json;
    report["benchmark"] = nlohmann::json::parse(outcome.nanobench_json);

    std::cout << report.dump(2) << "\n";
}

}  // namespace ort_runner
