# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-19

Rewritten from C++/CMake to Rust/Cargo, and **the tool now actually benchmarks** — 0.2.0 could
load a model and describe it, but printed `(benchmark not wired up yet)` when asked to measure
anything.

### Added

- **Benchmarking.** Warmup iterations followed by timed iterations, each inference timed
  individually. Reports count, mean, sample standard deviation, min, p50/p90/p95/p99 and max.
  Configure with `--warmup` and `--iterations`.
- **Memory attributable to the model**, decomposed into *weights* (session build: parameters plus
  the optimized graph, a fixed cost) and *working set* (inference: activation buffers, which scale
  with input shape). Read from `/proc/self/smaps_rollup` as PSS and private-dirty deltas between
  phases, plus a `getrusage` peak RSS.
- **JSON report**, written to `reports/` beside the executable after every run. Carries the
  command line, resolved configuration, device details, model self-description, statistics and
  every raw per-iteration sample — so a run can be re-analysed without re-running it on the
  device. The file embeds its own field guide, caveats and reporting guidance, because it is fed
  to a language model that writes the human-facing summary.
- **Human report** on stdout: the handful of numbers that decide something, coloured, with units
  scaled per magnitude (s/ms/µs/ns).
- **Real input tensors** via `--inputs model_inputs.npz`. The archive must supply every model
  input and nothing else; a mismatch fails with a message naming both sides.
- **Model self-description** in the report: `doc_string`, producer, version, custom
  `metadata_props` and file size — what identifies *which* model produced a number.
- **`examples/large_reduce`**, a benchmark fixture over a fully dynamic tensor, sized from the
  command line, that does enough work to measure.
- **`just fetch`** to populate the crate registry, and **`just test-e2e`** for the tests that need
  a real ONNX Runtime.

### Changed

- **Build system is Cargo**, driven through the same Podman toolchain images. Both Linux targets
  now build from one image (Ubuntu 18.04, glibc 2.25 — lower than 0.2.0's 2.31, so binaries reach
  further back). Android cross-compiles from an arm64 image against the NDK sysroot rather than
  running the NDK's x86_64 toolchain under emulation.
- **`--iterations` replaces `--min-epoch-iterations`.** Epoch-based timing averages inside a
  batch, which destroys the tail latency that decides whether a model is usable on a device. One
  inference is milliseconds, so per-iteration clock overhead is under a thousandth of a percent.
- **Warmup is timed and reported**, not discarded — it is the only evidence of first-inference
  cost — but excluded from the statistics so a cold start cannot appear as tail latency.
- **Percentiles are nearest-rank**, so every reported value is a latency that actually occurred
  rather than an interpolated estimate.
- **Execution provider availability is probed at runtime**, never assumed at compile time. The
  C++ decided this with `#ifdef __ANDROID__` and carried a comment claiming XNNPACK worked on
  Linux; the Linux prebuilts ship no XNNPACK kernels, so that flag could never have worked there.
- Run configuration is separated from the argv surface, so sessions can be built from generated
  configurations rather than only from a command line.

### Fixed

- **`just fmt` had never once run.** Every cargo subcommand was passed `--target`, which rustfmt
  rejects, so the recipe failed silently and the tree drifted out of format.
- **A hanging unit test could hang the whole suite indefinitely.** Constructing an `ort` tensor
  initialises ONNX Runtime, which blocks rather than erroring when no library is loaded. Tests
  that need a runtime are now separated and run by `just test-e2e`; every containerised command is
  bounded by a timeout and killed rather than waited on.
- **The release path was broken end to end** after the cargo migration: nothing bundled
  `libonnxruntime.so` beside the built binary, and the packaging step read the project version
  from the deleted `CMakeLists.txt`.

### Removed

- The C++ tree, CMake, the vendored headers and `~/.ccache`.
- `--min-epoch-iterations` (see above).

## [0.2.0] - 2026-07-09

### Added

- **Linux aarch64 target** (`linux-aarch64`), built for portability to devices like the
  Raspberry Pi 4: an old-glibc (debian bullseye, glibc 2.31) base plus statically linked
  libstdc++/libgcc, so the binary runs on any newer-glibc aarch64 system with only glibc as a
  runtime dependency.
- **Android armeabi-v7a target** (`android-armv7`, hardfloat), alongside the existing arm64-v8a
  build; both ABIs come from the official ONNX Runtime AAR.
- **Post-build smoke test** (`scripts/smoke.py`), run automatically after every build: verifies
  the binary's ELF architecture, that `libonnxruntime.so*` is bundled beside it, and (for Linux)
  that `ort_runner --version` runs and links against ONNX Runtime.

### Changed

- Build targets are now arch-explicit: `linux` → `linux-x64`, `android` → `android-arm64`, with
  `linux-aarch64` and `android-armv7` added. Each builds into its own `build-<target>/` directory
  and fetches into its own `sdk/onnxruntime-<target>/` directory (the two Linux arches and two
  Android ABIs no longer collide).
- Rewrote `scripts/fetch_onnxruntime.sh` as `scripts/fetch_onnxruntime.py` (standard library
  only). ONNX Runtime is always the official prebuilt binary — never built from source.
- Toolchain images obtain cmake and ninja from PyPI (in an isolated venv) so every target uses
  the same modern cmake regardless of the base distro's package age.

[0.3.0]: https://github.com/Red-Eyed/ort_runner/releases/tag/v0.3.0
[0.2.0]: https://github.com/Red-Eyed/ort_runner/releases/tag/v0.2.0
