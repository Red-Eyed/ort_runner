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
timing/statistics, JSON output) is deliberately built on popular MIT-licensed header-only
libraries (see [third_party/README.md](third_party/README.md)) rather than hand-rolled.

## Build

Requires [Podman](https://podman.io/); no local C++ toolchain or Android NDK needed -- both
build environments are containerized.

```bash
just build-linux      # native Linux (x86_64), into build-linux/
just build-android    # cross-compiled Android arm64-v8a, into build-android/
```

Each target fetches its own pinned ONNX Runtime distribution on first build (cached on the
host under `sdk/`, gitignored) via `scripts/fetch_onnxruntime.sh`. Compiles are cached via
`ccache`, persisted on the host under `.ccache/` across `podman run` invocations.

The Android toolchain image always targets `linux/amd64` (see the Justfile) regardless of the
build host's own architecture: the NDK only ships a Linux **host** toolchain for x86_64, so on
an arm64 host (e.g. Apple Silicon) it runs under QEMU emulation -- confirmed working, just
slower than a native x86_64 host.

## Run

```bash
# Native Linux
just run-linux path/to/model.onnx

# Android: pushes the binary + libonnxruntime.so to a connected device/emulator and runs it
just push-android path/to/model.onnx
```

Or invoke `ort_runner` directly once built:

```bash
LD_LIBRARY_PATH=build-linux/bin build-linux/bin/ort_runner --model path/to/model.onnx
```

Inspect a model's declared inputs/outputs without running inference:

```bash
ort_runner --model path/to/model.onnx --list-io
```

Full flag reference: `ort_runner --help`.

## Known limitations

- **Dynamic/symbolic input dimensions** are resolved by substituting `--default-dim` (default
  `1`) for any declared dim `<= 0` -- this is ORT's convention for both plain dynamic dims and
  named symbolic dims (both surface as `-1` from `GetShape()`). A fully unranked input (no
  shape info at all) is rejected with a clear error rather than guessed at.
- **Randomly-filled integer inputs** (e.g. embedding/gather indices) are clamped to
  `[0, --int-fill-max]` (default `15`) as a mitigation, not a fix -- a model expecting indices
  in a specific valid range may still throw on nonsensical synthesized input. This is an
  inherent limitation of generating inputs from metadata alone, with no real sample data.
- Only a subset of ONNX tensor element types is supported for auto-generation:
  float32/float64/int64/int32/int16/int8/uint8/bool. Others (float16, string, complex, ...)
  raise a clear error.
- Execution provider support is CPU (default, always available) and NNAPI (Android builds
  only, `--provider nnapi`). XNNPACK is not available -- it isn't included in ONNX Runtime's
  official prebuilt packages and would require a custom ORT build.
- Loading real input data from files (rather than auto-generating) is a reserved, not yet
  implemented, stretch goal (`--input name=path`).
