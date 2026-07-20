# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-07-20

### Added

- **`--iteration-timeout <seconds>`** (default `10`, `0` disables), bounding how long a single
  inference may take. An inference that never returns previously took the whole run with it —
  no statistics, no report, and nothing to say which stage was at fault. The limit is per
  inference rather than per run, because total runtime is already chosen through `--iterations`;
  a limit on the whole run would eventually cut short a long benchmark behaving exactly as asked.
  The value is recorded in the report's `bench_config`.

  Overruns are terminated through ONNX Runtime rather than by killing the process, so they
  surface as an ordinary error naming the flag that set the limit.

  It does **not** cover model loading or execution-provider initialisation, which happen before
  there is any inference to abandon. A provider that hangs while initialising its device — as
  `--provider webgpu` does with quantized models on Adreno hardware — is not caught by this.

### Changed

- **A model taking longer than 10 seconds per inference now fails by default** rather than
  running to completion. Raise the limit or pass `--iteration-timeout 0` to restore the previous
  behaviour.

## [0.4.1] - 2026-07-20

### Fixed

- **Abort at exit on Android**, after the run had completed and the report had been printed
  (`FORTIFY: pthread_mutex_lock called on destroyed mutex`). ONNX Runtime requires `ReleaseEnv` to
  precede the static destructors in `libonnxruntime.so`; the C++ implementation retired in 0.3.0
  guaranteed that by destroying its `Ort::Env` at the end of `main`, whereas the `ort` crate must
  keep the environment alive for the process and defers `ReleaseEnv` to a `.fini_array` handler
  that runs alongside those destructors. The process now flushes its output and exits immediately
  once the report is written, so that phase is never entered. Only bionic checks for this, which is
  why Linux never showed it.

  Confirmed fixed on a device. The diagnosis behind it is drawn from the ONNX Runtime and `ort`
  sources rather than from a backtrace, so it remains the best available explanation of *why* the
  fix works; and the fix is a mitigation in this tool rather than a cure in `ort`.

## [0.4.0] - 2026-07-20

### Changed

- **`--profile` writes to `ort_profiler/` beside the executable**, a sibling of `reports/`, so a
  profiled run's two artifacts come off a device together. The path is now printed after the run
  and recorded in the JSON report as `profile_path`. Previously the trace was written relative to
  the process working directory, which under `adb shell` is `/` and not writable — so on a device,
  the case this tool exists for, the trace was lost.
- **`--profile-prefix` removed.** The trace is named after the model, matching how `reports/`
  names its files.

### Fixed

- **`--profile` produced an empty trace.** ONNX Runtime only flushes the profile when profiling is
  explicitly ended, which never happened.
- **`just download-prebuilt` with no targets named** now downloads every target, as its help
  always claimed. Its `--all` flag never worked: `argparse` applies `choices` to the empty list a
  variadic positional yields, so naming no targets failed with `invalid choice: []`. The flag is
  gone; naming nothing means everything.

## [0.3.0] - 2026-07-19

Rewritten from C++/CMake to Rust/Cargo. Everything 0.2.0 benchmarked, this benchmarks — what
changed is how the measurement is taken, what it reports about memory, and how you get the binary.

### Added

- **Memory attributable to the model**, decomposed into *weights* (session build: parameters plus
  the optimized graph, a fixed cost) and *working set* (inference: activation buffers, which scale
  with input shape). Read from `/proc/self/smaps_rollup` as PSS and private-dirty deltas between
  phases. 0.2.0 reported a single peak RSS, which counts shared library pages in full and is
  therefore near-identical for every model — it cannot distinguish two of them.
- **`--dim <name>=<value>`** to size one named symbolic dimension. 0.2.0 had only `--default-dim`,
  which applies to every dynamic axis at once, so a model with independent axes could not be given
  a realistic shape.
- **`--iterations`**, replacing `--min-epoch-iterations`.
- **A JSON report written after every run** to `reports/` beside the executable, carrying the
  command line, resolved configuration, device, model self-description, statistics and every raw
  per-iteration sample. 0.2.0 wrote JSON to stdout only when asked. The file embeds its own field
  guide, caveats and reporting guidance, because it is fed to a language model that writes the
  human-facing summary.
- **Model self-description** in the report: `doc_string`, producer, version, custom
  `metadata_props` and file size — what identifies *which* model produced a number.
- **`android-x86_64`** (emulator ABI) as a released target.
- **`download_prebuilt.py` and prebuilt-first `run-*` recipes.** Running a release needs only
  Python 3.9+ (standard library) and, for a phone, `adb` — no Podman, Rust or NDK. Building from
  source moves to `run-dev-*`.
- **`examples/large_reduce`**, a fixture over a fully dynamic tensor, sized from the command line,
  that does enough work to measure.
- **`just fetch`** to populate the crate registry, and **`just test-e2e`** for the tests that need
  a real ONNX Runtime.

### Changed

- **Build system is Cargo**, driven through the same Podman toolchain images. Both Linux targets
  build from one image (Ubuntu 18.04, glibc **2.25** — down from 0.2.0's 2.31, so binaries reach
  further back). Android cross-compiles from an arm64 image against the NDK sysroot rather than
  running the NDK's x86_64 toolchain under emulation.
- **No more epochs.** 0.2.0 timed batches of iterations through nanobench and reported per-epoch
  averages; every inference is now timed individually. Averaging inside a batch destroys the tail
  latency that decides whether a model is usable on a device, and at millisecond scale the
  per-iteration clock overhead it was avoiding is under a thousandth of a percent.
- **Warmup timings are recorded and reported**, separately from the statistics. 0.2.0 counted
  warmup iterations but discarded their durations; they are the only evidence of what the first
  inference costs, which is what matters for anything run once per user action.
- **Percentiles are nearest-rank**, so every reported value is a latency that actually occurred.
  0.2.0 interpolated between neighbouring samples, which can report a latency no iteration took.
- **Execution provider availability is probed at runtime.** 0.2.0 decided it at compile time with
  `#ifdef __ANDROID__`, and carried a comment claiming XNNPACK worked on Linux; the Linux prebuilts
  ship no XNNPACK kernels, so that flag could never have worked there.
- **`--info`** reports device, host and runtime details, including which providers the loaded
  runtime actually contains.
- Run configuration is separated from the argv surface, so sessions can be built from generated
  configurations rather than only from a command line.
- Rust edition 2024.

### Fixed

- **`just fmt` had never once run.** Every cargo subcommand was passed `--target`, which rustfmt
  rejects, so the recipe failed silently and the tree drifted out of format.
- **A test could hang the suite indefinitely.** Constructing an `ort` tensor initialises ONNX
  Runtime, which blocks rather than erroring when no library is loaded. Tests needing a runtime are
  now separated behind `just test-e2e`, and every containerised command is bounded by a timeout and
  killed rather than waited on.
- **The release path was broken end to end** after the cargo migration: nothing bundled
  `libonnxruntime.so` beside the built binary, packaging read the version from the deleted
  `CMakeLists.txt`, and `build-all` covered four of the five targets it packages.

### Removed

- The C++ tree, CMake, the vendored headers (nanobench, magic_enum, json) and `~/.ccache`.
- `--min-epoch-iterations`, superseded by `--iterations`.

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

[0.5.0]: https://github.com/Red-Eyed/ort_runner/releases/tag/v0.5.0
[0.4.1]: https://github.com/Red-Eyed/ort_runner/releases/tag/v0.4.1
[0.4.0]: https://github.com/Red-Eyed/ort_runner/releases/tag/v0.4.0
[0.3.0]: https://github.com/Red-Eyed/ort_runner/releases/tag/v0.3.0
[0.2.0]: https://github.com/Red-Eyed/ort_runner/releases/tag/v0.2.0
