#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace ort_runner {

enum class ExecutionProvider { kCpu, kNnapi };
enum class FillStrategy { kRandom, kOnes, kZeros };
enum class OutputFormat { kHuman, kJson };

struct Config {
    std::string model_path;
    uint64_t warmup_iterations = 3;
    uint64_t min_epoch_iterations = 1;
    std::optional<int> intra_op_threads;  // unset means "leave at onnxruntime's own default"
    int inter_op_threads = 1;
    ExecutionProvider provider = ExecutionProvider::kCpu;
    int64_t default_dim = 1;
    FillStrategy fill = FillStrategy::kRandom;
    uint64_t seed = 42;
    int64_t int_fill_max = 15;
    OutputFormat output_format = OutputFormat::kHuman;
    bool list_io = false;
};

// Parses argv into a Config. Returns std::nullopt if the process should exit with a failure
// code (invalid arguments); -h/--version are handled internally by argparse and exit(0)
// before this function returns.
std::optional<Config> ParseArgs(int argc, const char *const argv[]);

std::string ToString(ExecutionProvider provider);
std::string ToString(FillStrategy fill);
std::string ToString(OutputFormat format);

}  // namespace ort_runner
