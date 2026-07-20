# ort_runner

**Benchmark an ONNX model on a real device without writing any setup code.**

Point it at a `.onnx` file. It reads the model's declared shapes and dtypes, generates the input
tensors itself, runs inference, and reports the latency distribution and how much memory the model
is actually responsible for.

```console
$ ort_runner --model model.onnx

run
  model       model.onnx
  provider    cpu
  load        19.81 ms
  inputs      synthesized (generated values, not real data)

latency over 100 iterations
  p50         307.7 µs
  p90         338.1 µs
  p95         341.7 µs
  p99         363.6 µs
  mean        315.9 µs
  std dev     16.2 µs
  min / max   298.7 µs / 363.6 µs
  warmup      1.06 ms first, 574.7 µs mean over 3 (excluded)

memory attributable to this model
  weights     16.9 MiB pss, 7.0 MiB private dirty
  working set 16.4 MiB pss, 16.0 MiB private dirty
  peak rss    43.9 MiB

report: reports/model-20260719T180543Z.json
```

No Python on the device, no runtime to install: one binary and one `.so`, `adb push`ed and run.

## Why you might want it

**You have a model and no data.** The usual way to get a latency number is to write a harness that
fabricates plausible inputs, which means knowing every shape, dtype and dynamic axis. This derives
all of it from the model's own declarations. Auto-generated inputs are the point of the tool, not
a convenience feature.

**You need to know whether it fits, not just whether it's fast.** Most tools report peak RSS,
which counts shared library pages in full — so it is roughly the same for every model you load and
cannot tell two of them apart. `ort_runner` reports the *delta* the model is responsible for,
split into two numbers that call for opposite fixes:

| | what it is | how to shrink it |
|---|---|---|
| **weights** | parameters + optimized graph; fixed at export time | quantize, prune, smaller architecture |
| **working set** | activation buffers; scales with input shape | smaller batch, shorter sequence, different EP |

It reports PSS and private-dirty — what Android attributes to an app in `dumpsys meminfo`, and
what predicts a low-memory kill — rather than only RSS.

**You care about the tail.** Every inference is timed individually; there is no epoch averaging to
smear slow iterations away. Percentiles are **nearest-rank**, so a reported p99 is a latency that
actually occurred rather than an interpolated estimate. Warmup is timed and reported but excluded
from the statistics, so a cold start never masquerades as tail latency.

**Every run archives itself.** A JSON report lands in `reports/` with the command line, resolved
configuration, device details, the model's own metadata, and every raw per-iteration sample.
Getting a device back into the state that produced a number is the expensive part of benchmarking,
so the samples are kept and tomorrow's question doesn't cost another run.

That JSON also **describes itself** — it embeds a field guide, caveats and reporting guidance — so
it can be handed to a language model to write up without it having to guess what `p99_ms` covers.

## Install

**You do not need to build this.** Releases ship self-contained binaries; the only requirement is
**Python 3.9+** (standard library only — no pip install, no virtualenv), plus `adb` if you are
benchmarking a phone.

Grab everything and run it in one step:

```bash
just run-android-arm64 model.onnx --dim batch=1
```

That downloads the release for the target, pushes it to the connected device, and runs it. No
Podman, no Rust, no NDK. The download is idempotent, so later runs skip it.

No `just` either? The scripts are plain Python:

```bash
python3 scripts/download_prebuilt.py android-arm64
python3 scripts/run_android.py --source prebuilt android-arm64 model.onnx
```

Or skip the repo entirely — unpack a release zip and run the binary directly:

```bash
unzip ort_runner-v0.4.1-linux-aarch64-ort1.27.0.zip
cd    ort_runner-v0.4.1-linux-aarch64-ort1.27.0
./ort_runner --model model.onnx
```

Targets: `linux-x64`, `linux-aarch64`, `android-arm64`, `android-armv7`, `android-x86_64`.

A Linux binary is only run where it can execute *natively*. Timing one under emulation measures
the emulator rather than the model while looking exactly like a real result, so that combination
is refused rather than quietly reported.

Linux builds have a **glibc 2.25 floor** — low enough for a Raspberry Pi on Buster, and the same
floor ONNX Runtime's own prebuilts use, so the binary runs anywhere the runtime does.

## Try it in 30 seconds

No model handy? Generate one:

```bash
uv run examples/large_reduce/make_model.py
ort_runner --model examples/large_reduce/large_reduce.onnx \
           --dim batch=1 --dim height=2048 --dim width=2048
```

Every axis of that model is dynamic, so `--dim` sizes the run — halve the axes and watch the
working set fall with them.

## Usage

**Look at the model first.** Negative dimensions are the ones the model left open:

```console
$ ort_runner --model model.onnx --list-io
inputs:
  data: shape=[-1, -1, -1] dtype=float32 symbolic_dims=batch, height, width
outputs:
  mean: shape=[] dtype=float32
```

**Pin the dynamic dimensions.** `--dim` targets one named axis, `--default-dim` covers everything
else (default `1`). A `--dim` naming an axis the model doesn't have warns rather than failing, so
a typo can't silently do nothing:

```bash
ort_runner --model model.onnx --dim batch=1 --dim seq_len=128
ort_runner --model model.onnx --default-dim 224
```

**Feed real data** when the model's behaviour depends on input *values* — early exit, sparsity,
content-driven sequence length. One array per input, keyed by the input's name:

```python
np.savez("inputs.npz", input_ids=ids, attention_mask=mask)
```
```bash
ort_runner --model model.onnx --inputs inputs.npz
```

The archive must supply every input and nothing else. A mismatch is an error naming both sides,
never a fallback to synthesis: a typo'd key would otherwise benchmark generated data you believed
you had replaced, which is the one failure that is wrong without looking wrong.

**Choose an execution provider**, and see which ones your runtime actually contains:

```bash
ort_runner --info                                  # device, runtime, available providers
ort_runner --model model.onnx --provider nnapi
```

Availability is probed at runtime, never assumed at compile time, so this reports what your build
genuinely supports rather than what it was expected to.

**Tune threading.** Unset means ONNX Runtime's own default, which is not the same as any specific
number:

```bash
ort_runner --model model.onnx --threads 4 --inter-op-threads 1
ort_runner --model model.onnx --threads 4 --intra-op-spinning off   # less CPU burn while idle
```

`--threads` (intra-op) matters most for a single-graph CPU-bound model. `--intra-op-spinning off`
trades a little wake-up latency for less battery burn between ops — often the right trade on a
phone.

**Measure the true memory floor.** The CPU arena pre-allocates, so it reflects an allocation
strategy as much as the model's real demand. When memory is what you're measuring:

```bash
ort_runner --model model.onnx --disable-cpu-arena --disable-mem-pattern
```

**Find the slow op** with ONNX Runtime's own per-op profiler. The trace lands in `ort_profiler/`
beside the binary — alongside `reports/`, so both come off a device in one `adb pull` — and its
path is printed after the run and recorded in the report as `profile_path`. The output is
Chrome-trace JSON: open it at `chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev):

```bash
ort_runner --model model.onnx --profile
```

**Run on a device.** This downloads the release if needed, pushes the binary and
`libonnxruntime.so` to `/data/local/tmp/`, and runs it there over `adb shell`:

```bash
adb devices
just run-android-arm64 model.onnx --dim batch=1 --provider nnapi
```

`--help` lists the rest: graph optimization level, memory-pattern control, fill strategy and seed
for generated inputs, log severity, and dumping the post-optimization graph.

## Build from source

Requires only [Podman](https://podman.io/) and [just](https://github.com/casey/just). No Rust
toolchain, no Android NDK, no C++ compiler on your machine — every build runs in a container.

```bash
just build-linux-aarch64     # or linux-x64, android-arm64, android-armv7, android-x86_64
just test                    # unit tests; needs no ONNX Runtime at all
just check                   # clippy + unit tests + runtime-gated tests
just run-dev-android-arm64 model.onnx    # run what you just built, not a release
```

The `run-*` recipes deliberately use released binaries; `run-dev-*` uses your local build. Which
binary produced a measurement is never left implicit — a stale download beside a fresh build would
otherwise attribute numbers to the wrong one.

Everything above lives in `dev.just`, leaving the Justfile itself to the `run-*` recipes that need
no toolchain. It is imported rather than namespaced, so recipes are invoked the same either way.

The pinned ONNX Runtime is fetched automatically on first build, and is always Microsoft's
official prebuilt binary — never built from source.

Android cross-compiles from an arm64 image against the NDK sysroot rather than emulating the NDK's
own x86_64 toolchain, because `rustc` segfaults under QEMU.

## Honest limitations

- **A demo and experimentation project**, not a published or battle-tested tool.
- **There is no CI**; releases are cut from a developer machine with `just release`. Android
  binaries are checked for ELF architecture and bundling at build time, but on-device runs are
  manual.
- **Memory figures cover this process only.** Under `--provider nnapi` or `webgpu`, allocations
  happen in a vendor HAL or GPU driver where `/proc` cannot see them; the numbers are complete for
  `cpu` and `xnnpack`.
- **Synthesized inputs are type- and shape-correct, not representative.** Random integer inputs
  are clamped to `[0, --int-fill-max]` (default 15) because index inputs blow up on out-of-range
  values — a mitigation, not a fix, since only you know the real vocabulary size. When it bites,
  use `--inputs`.
- **A subset of element types** is supported for generation: float32/float64, int64/int32/int16/
  int8, uint8, bool. Others (float16, string, complex) raise a clear error.
- **A fully unranked input is rejected** rather than guessed at: with no rank there is no
  dimension count to synthesize against, and no flag can supply one.
- **XNNPACK is compiled into the official prebuilt `.so`s but gated off at runtime.** Symbol
  presence is not availability — ORT hides it behind a build-time capability flag the release
  artifacts don't set. It needs a custom ONNX Runtime build.
- **No 32-bit ARM Linux target** — ONNX Runtime ships no prebuilt for it. 32-bit ARM is covered on
  Android, where the AAR does.

## License

MIT — see [LICENSE](LICENSE).
