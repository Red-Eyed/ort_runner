#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace ort_runner {

// Enum members are deliberately lowercase, matching their CLI flag values exactly (rather
// than the usual kPascalCase), so magic_enum's enum_name()/enum_cast<T>() round-trip them to
// and from argv without any hand-written string tables.
enum class ExecutionProvider { cpu, nnapi, xnnpack };
enum class FillStrategy { random, ones, zeros };
enum class OutputFormat { human, json };
enum class ExecutionMode { sequential, parallel };
enum class GraphOptimizationLevel { disable, basic, extended, layout, all };
enum class LogSeverity { verbose, info, warning, error, fatal };

// Per-symbolic-name overrides for dynamic input dimensions, e.g. {"batch": 4, "seq_len": 128}.
using DimOverrides = std::unordered_map<std::string, int64_t>;

struct Config {
    std::string model_path;
    // Path to a .npz of named input arrays (numpy.savez); array names must match model input
    // names. Inputs absent from the archive are synthesized. Unset means synthesize everything.
    std::optional<std::string> inputs_path;
    uint64_t warmup_iterations = 3;
    uint64_t min_epoch_iterations = 1;
    std::optional<int> intra_op_threads;  // unset means "leave at onnxruntime's own default"
    int inter_op_threads = 1;
    ExecutionMode execution_mode = ExecutionMode::sequential;
    std::optional<bool> intra_op_spinning;  // unset means "leave at onnxruntime's own default"
    std::optional<bool> inter_op_spinning;  // unset means "leave at onnxruntime's own default"
    ExecutionProvider provider = ExecutionProvider::cpu;
    int64_t default_dim = 1;
    DimOverrides dim_overrides;
    FillStrategy fill = FillStrategy::random;
    uint64_t seed = 42;
    int64_t int_fill_max = 15;
    OutputFormat output_format = OutputFormat::human;
    bool list_io = false;
    bool profile = false;
    std::string profile_prefix = "ort_runner_profile";
    GraphOptimizationLevel graph_optimization_level = GraphOptimizationLevel::all;
    bool disable_cpu_arena = false;
    bool disable_mem_pattern = false;
    std::optional<std::string> optimized_model_path;
    LogSeverity log_severity = LogSeverity::warning;
};

// Parses argv into a Config. Returns std::nullopt if the process should exit with a failure
// code (invalid arguments); -h/--version are handled internally by argparse and exit(0)
// before this function returns.
std::optional<Config> ParseArgs(int argc, const char *const argv[]);

}  // namespace ort_runner
