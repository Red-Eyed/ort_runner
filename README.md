# ort_runner

A console app for benchmarking ONNX Runtime inference on Linux and Android, modeled
conceptually on ExecuTorch's
[`executor_runner.cpp`](https://github.com/pytorch/executorch/blob/main/examples/portable/executor_runner/executor_runner.cpp).
Point it at a `.onnx` model; it auto-generates input tensors from the model's own declared
shapes/dtypes (no sample data required), runs the model, and reports latency and peak memory.

**This is a demo/experimentation project, not a published or production-ready tool.** Built to
work through cross-compiling a small C++ CLI against ONNX Runtime for on-device (Android)
benchmarking, containerized via Podman.

## Why this exists

The use case: copy a single prebuilt executable onto a device (`adb push`), point it at a
model path, and run it -- no need to hand-craft representative input tensors just to get a
latency number. Input auto-generation is the core feature; everything else (CLI parsing,
timing/statistics, JSON output, enum-string conversion) is deliberately built on popular
MIT-licensed header-only libraries (see [third_party/README.md](third_party/README.md))
rather than hand-rolled.

## Build

Requires [Podman](https://podman.io/); no local C++ toolchain or Android NDK needed -- both
build environments are containerized.

```bash
just build-linux      # native Linux (x86_64), into build-linux/
just build-android    # cross-compiled Android arm64-v8a, into build-android/
```

Each target fetches its own pinned ONNX Runtime distribution on first build (cached on the
host under `sdk/`, gitignored) via `scripts/fetch_onnxruntime.sh`. Compiles are cached via
`ccache`, persisted across `podman run` invocations in `~/.ccache` on the host (created
automatically on the first build).

The Android toolchain image always targets `linux/amd64` (see `scripts/targets.py`) regardless of the
build host's own architecture: the NDK only ships a Linux **host** toolchain for x86_64, so on
an arm64 host (e.g. Apple Silicon) it runs under QEMU emulation -- confirmed working, just
slower than a native x86_64 host.

## Run

```bash
# Native Linux
just run-linux path/to/model.onnx

# Android: pushes the binary + libonnxruntime.so to a connected device/emulator and runs it
just run-android path/to/model.onnx
```

`build-linux/bin/` is self-contained (the binary carries a `$ORIGIN`-relative rpath to the
`libonnxruntime.so.1` copied beside it), so on a native Linux host you can also invoke it
directly, or copy `build-linux/bin/` elsewhere and run it from there, without
`LD_LIBRARY_PATH`:

```bash
build-linux/bin/ort_runner --model path/to/model.onnx
```

Inspect a model's declared inputs/outputs without running inference:

```bash
ort_runner --model path/to/model.onnx --list-io
```

Full flag reference: `ort_runner --help`.

## Usage guide

### 1. Inspect the model first

Always start with `--list-io` -- it shows exactly what the model expects before you run
anything, including which dimensions are dynamic and (if named) what they're called:

```
$ ort_runner --model model.onnx --list-io
inputs:
  x: shape=[1, 3, -1, -1] dtype=float32 symbolic_dims=H, W
  idx: shape=[-1] dtype=int64 symbolic_dims=N
outputs:
  y: shape=[1, 3, -1, -1] dtype=float32
  idx_out: shape=[-1] dtype=int64
```

`-1` marks a dynamic dimension. `symbolic_dims` names the ones the graph gave a name to --
these are exactly the names `--dim` (below) can target. A dynamic dim with no name (blank in
`symbolic_dims`) is anonymous and can only be controlled via `--default-dim`.

### 2. Run a basic benchmark

```
$ ort_runner --model model.onnx
=== ort_runner ===
timestamp:          2026-07-08T14:32:07Z
host:               Linux 6.6.87.2-microsoft-standard-WSL2 (x86_64), 8 cores
model:              model.onnx (4.2 MB)
ort_version:        1.27.0
provider:           cpu
...
load_time:          19.146 ms

benchmark: inference
  warmup_runs (skipped, unmeasured):  3
  measured_runs (epochs):             612
  total_iterations:                   612
  total_measured_time:                0.521 s
  throughput:                         1176.62 inferences/sec

  latency_ms:
    count         612.000
    mean            0.850
    std             0.042
    min             0.781
    25%             0.821
    50%             0.847
    75%             0.876
    max             1.204

peak_rss:     24924 KB (24.3 MB)
```

With no other flags: random inputs sized `1` for every dynamic dimension, CPU execution,
ORT's own default thread/session settings. Good for "does this model run at all," not yet a
meaningful latency number for most real models.

### 3. Set realistic shapes for dynamic dimensions

Anonymous dims (or ones you don't care to target individually) go through `--default-dim`;
named ones you *do* care about go through `--dim`, which takes priority:

```
$ ort_runner --model model.onnx --dim H=32 --dim W=32 --default-dim 5
...
inputs (post dynamic-dim substitution, default_dim=5):
  x: declared=[1, 3, -1, -1] resolved=[1, 3, 32, 32] dtype=float32 symbolic_dims=H, W
  idx: declared=[-1] resolved=[5] dtype=int64 symbolic_dims=N
```

`H`/`W` resolved to the values you asked for; `N` (not targeted) fell back to
`--default-dim 5`. A `--dim` name that matches nothing in the model prints a stderr warning
instead of silently doing nothing -- catches typos in the name.

### 4. Compare execution providers

```bash
ort_runner --model model.onnx --provider cpu       # always available, the safe baseline
ort_runner --model model.onnx --provider nnapi      # Android builds only
ort_runner --model model.onnx --provider xnnpack    # see the note below
```

Compare the `latency_ms` mean/`throughput` lines across runs to see which is actually faster for a given model
on a given device -- there's no universal winner. Note: `xnnpack` is implemented correctly but
throws `XNNPACK execution provider is not supported in this build` against the official
prebuilt ONNX Runtime packages this tool fetches by default (see Configuration below for why);
it only works if you point `ORT_RUNNER_SDK_DIR` at a custom ORT build compiled with XNNPACK
enabled.

### 5. Tune threading for the target device

```bash
ort_runner --model model.onnx --threads 4 --inter-op-threads 1
ort_runner --model model.onnx --threads 4 --intra-op-spinning off  # lower CPU/battery use
```

`--threads` (intra-op) is the one that matters most for a single-graph CPU-bound model.
`--intra-op-spinning off` trades a little wake-up latency for less CPU burn while idle between
ops -- worth trying on a battery-powered device where thermals/battery matter more than
shaving the last microseconds off latency.

### 6. Get an accurate memory reading

The CPU memory arena pre-allocates a chunk up front, which can make `peak_rss` reflect the
arena's allocation strategy more than the model's actual working set. If memory budget is
what you're actually measuring (not speed), disable it for a truer number:

```bash
ort_runner --model model.onnx --disable-cpu-arena --disable-mem-pattern
```

### 7. Profile a model to find slow ops

```
$ ort_runner --model model.onnx --profile --profile-prefix myrun
...
profile:      myrun_2026-07-06_18-30-57_276.json
```

The written file is a standard Chrome-trace-format JSON (open it at `chrome://tracing` or
`https://ui.perfetto.dev`, or process it directly -- it's a flat JSON array of `{name, dur,
ts, ...}` events, one per op per iteration) showing per-op timing, useful for finding which
specific op in the graph dominates latency rather than just the end-to-end number.

### 8. Machine-readable output for scripts/CI

```bash
ort_runner --model model.onnx --output-format json > result.json
jq '.peak_rss_kb, .load_time_ms, .provider' result.json
jq '.benchmark.results[0].measurements | map(.elapsed) | add / length' result.json
```

`--output-format json` is the *only* thing written to stdout in that mode (no preamble text
mixed in), so it's always safe to pipe directly -- one merged document combining this tool's
own fields with nanobench's own per-epoch measurements under `.benchmark`.

### 9. Benchmark on an actual Android device

```bash
adb devices                    # confirm a device/emulator is connected first
just run-android model.onnx -- --dim H=32 --dim W=32 --provider nnapi
```

This builds (if needed), pushes the binary + `libonnxruntime.so` to `/data/local/tmp/`, and
runs it there over `adb shell`. Any flags after `--` pass straight through to `ort_runner`.

### Troubleshooting

| Message | What it means |
|---|---|
| `input 'X' has element type ..., which is outside the subset auto-generation supports` | The model has an input type auto-generation doesn't handle (float16, string, complex, ...) -- see Known limitations. |
| `input 'X' has no shape information at all (fully unranked)` | The model doesn't even declare a rank for that input; there's no dimension count to synthesize against. |
| `--dim <name> does not match any symbolic dimension name in this model's inputs` (stderr warning, not fatal) | Typo in the name, or the axis you meant is actually anonymous -- check `--list-io`'s `symbolic_dims` output. |
| `XNNPACK execution provider is not supported in this build` | Expected against the official prebuilt ORT packages (see `--provider` in Configuration) -- not a bug in this tool. |
| `--provider nnapi is only supported in Android builds` | Self-explanatory; only meaningful when built via the `android-arm64` preset. |
| A model expecting valid categorical/index values crashes on nonsensical input | Inherent to metadata-only auto-generation -- try `--int-fill-max` tuned to the model's real vocab/index size, or `--fill zeros`/`--fill ones` as a sanity check. |

## Configuration

Beyond the basics (`--model`, `--warmup`, `--min-epoch-iterations`, `--fill`, `--seed`,
`--output-format`, `--list-io`):

- **Dynamic/symbolic input dimensions**: `--default-dim <int>` (default `1`) covers any
  dynamic dim not otherwise named. `--dim <name>=<value>` (repeatable) overrides one specific
  *named* symbolic dimension, e.g. `--dim batch=4 --dim seq_len=128` -- run `--list-io` first
  to see each input's `symbolic_dims`. An override for a name that appears nowhere in the
  model prints a stderr warning (typo protection), not a hard error.
- **Threading**: `--threads`/`--inter-op-threads` (intra/inter-op thread counts),
  `--execution-mode sequential|parallel`, `--intra-op-spinning`/`--inter-op-spinning on|off`
  (thread spin-vs-sleep behavior while waiting for work).
- **Execution provider**: `--provider cpu|nnapi|xnnpack`. `nnapi` is Android-only and is a
  deprecated Android API (Google has been steering away from it since Android 15) but still
  functional on existing devices. `xnnpack` is registered via ONNX Runtime's newer generic
  `AppendExecutionProvider("XNNPACK")` API (not a dedicated per-provider function) rather than
  being available on every build: the `xnnpack::*` kernel code is compiled into **both**
  official prebuilt `.so`s (confirmed via `strings`), but empirically it still throws
  `XNNPACK execution provider is not supported in this build` at runtime against the official
  prebuilt packages -- ORT gates it behind an additional build-time capability flag that isn't
  set for the standard release artifacts. Symbol presence alone was not sufficient evidence
  (a lesson learned mid-implementation); this flag only becomes usable against a custom ONNX
  Runtime build compiled with XNNPACK support enabled, swapped in via `ORT_RUNNER_SDK_DIR`.
- **Profiling**: `--profile` enables ONNX Runtime's built-in per-op profiler
  (`--profile-prefix` sets the output file prefix); the written Chrome-trace-format JSON
  file's path is printed after the run.
- **Session tuning**: `--graph-optimization-level disable|basic|extended|layout|all` (default
  `all`, ORT's own session default), `--disable-cpu-arena`/`--disable-mem-pattern` (relevant
  for accurate peak-RSS measurement, since the arena pre-allocates), `--optimized-model-path
  <path>` (dumps the post-optimization graph), `--log-severity
  verbose|info|warning|error|fatal`.

## Known limitations

- **Randomly-filled integer inputs** (e.g. embedding/gather indices) are clamped to
  `[0, --int-fill-max]` (default `15`) as a mitigation, not a fix -- a model expecting indices
  in a specific valid range may still throw on nonsensical synthesized input. This is an
  inherent limitation of generating inputs from metadata alone, with no real sample data.
- Only a subset of ONNX tensor element types is supported for auto-generation:
  float32/float64/int64/int32/int16/int8/uint8/bool. Others (float16, string, complex, ...)
  raise a clear error.
- A fully unranked input (no shape info at all, not even a rank) is rejected with a clear
  error rather than guessed at -- there's no `--dim`/`--default-dim` override for a dimension
  count that isn't known at all.
- Loading real input data from files (rather than auto-generating) is a reserved, not yet
  implemented, stretch goal (`--input name=path`).

## License

MIT -- see [LICENSE](LICENSE). Vendored third-party headers keep their own licenses (all MIT;
see [third_party/README.md](third_party/README.md)).
