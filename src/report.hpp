#pragma once

#include <optional>
#include <string>
#include <vector>

#include "benchmark.hpp"
#include "config.hpp"
#include "input_synth.hpp"

namespace ort_runner {

// --list-io: prints declared input/output names, shapes (dynamic dims shown as-is, no
// substitution), and dtypes. No benchmark involved.
void PrintIoDescription(const std::vector<InputSpec> &inputs,
                         const std::vector<OutputSpec> &outputs);

// Human-readable preamble printed before the benchmark runs: model path, ORT version,
// provider, thread config, fill strategy/seed, load time, and resolved input/output shapes
// (important since input auto-generation is the whole point -- the user needs to see what
// was substituted for any dynamic dimension).
void PrintPreamble(const Config &config, double load_time_ms,
                    const std::vector<InputSpec> &inputs,
                    const std::vector<OutputSpec> &outputs);

// Human-readable summary of the benchmark run: warmup/measured run counts and a
// pandas-df.describe()-style breakdown of per-epoch latency (ms).
void PrintBenchmarkSummary(const BenchmarkStats &stats);

// Human-readable trailer, printed after PrintBenchmarkSummary(). profile_file is set only
// when --profile was passed.
void PrintTrailer(long peak_rss_kb, const std::optional<std::string> &profile_file);

// Single merged JSON document: nanobench's own rendered JSON plus this tool's fields, since
// nanobench's JSON template has no way to know about them. The only stdout output in JSON mode.
void PrintJsonReport(const Config &config, double load_time_ms,
                      const std::vector<InputSpec> &inputs,
                      const std::vector<OutputSpec> &outputs, const BenchmarkOutcome &outcome,
                      const std::optional<std::string> &profile_file);

}  // namespace ort_runner
