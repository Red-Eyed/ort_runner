# Every recipe runs inside a podman toolchain image -- nothing is built, tested or linted on
# the host. Two directories in the bind-mounted repo make that cheap rather than repetitive:
#
#   .cargo/   CARGO_HOME       -- the crate registry, so dependencies download once
#   target/   CARGO_TARGET_DIR -- compiled artifacts, so only changed crates rebuild
#
# Both are gitignored and persist across `podman run` invocations, which is what the C++ build
# used ~/.ccache for. Artifacts are namespaced per target triple, so the four targets share one
# directory without clobbering each other.

# Target used for tests and lints: the one that runs natively on an Apple Silicon host, since
# these run often and emulation is the slowest part of any container build.
dev_target := "linux-aarch64"

default:
    @just --list

# --- images -----------------------------------------------------------------

image-linux-x64:
    uv run scripts/build_image.py linux-x64

image-linux-aarch64:
    uv run scripts/build_image.py linux-aarch64

# Shared by both Android ABIs.
image-android:
    uv run scripts/build_image.py android-arm64

images: image-linux-x64 image-linux-aarch64 image-android

# --- build ------------------------------------------------------------------

build-linux-x64: image-linux-x64
    uv run scripts/build.py linux-x64
    uv run scripts/smoke.py linux-x64

build-linux-aarch64: image-linux-aarch64
    uv run scripts/build.py linux-aarch64
    uv run scripts/smoke.py linux-aarch64

build-android-arm64: image-android
    uv run scripts/build.py android-arm64
    uv run scripts/smoke.py android-arm64

build-android-armv7: image-android
    uv run scripts/build.py android-armv7
    uv run scripts/smoke.py android-armv7

build-all: build-linux-x64 build-linux-aarch64 build-android-arm64 build-android-armv7

# --- dependencies -----------------------------------------------------------

# Populates .cargo/registry so every containerised cargo invocation can stay --offline.
#
# Runs on the host, not in an image, for the same reason SDK fetching does: cargo's network
# path hangs indefinitely inside these containers. Run it after changing Cargo.toml; builds and
# tests need no network once it has.
fetch:
    CARGO_HOME={{justfile_directory()}}/.cargo cargo fetch --locked

# --- test and lint ----------------------------------------------------------

# Unit tests. The pure-logic modules need no ONNX Runtime, so this needs no fetched SDK.
test: (_image dev_target)
    uv run scripts/cargo.py {{dev_target}} test

lint: (_image dev_target)
    uv run scripts/cargo.py {{dev_target}} clippy --all-targets

fmt: (_image dev_target)
    uv run scripts/cargo.py {{dev_target}} fmt

# Integration tests that build a real ort Tensor, so unlike `test` these need a runtime.
# Fetches the SDK if it is missing.
test-e2e: (_image dev_target)
    uv run scripts/test_e2e.py {{dev_target}}

# Everything CI would check.
check: lint test test-e2e

_image target:
    uv run scripts/build_image.py {{target}}

# --- run --------------------------------------------------------------------

run-linux-x64 model *args: build-linux-x64
    uv run scripts/run_linux.py linux-x64 {{model}} {{args}}

run-linux-aarch64 model *args: build-linux-aarch64
    uv run scripts/run_linux.py linux-aarch64 {{model}} {{args}}

# Pushes the binary + libonnxruntime.so to a connected device and runs it there.
run-android-arm64 model *args: build-android-arm64
    uv run scripts/run_android.py android-arm64 {{model}} {{args}}

run-android-armv7 model *args: build-android-armv7
    uv run scripts/run_android.py android-armv7 {{model}} {{args}}

# --- package ----------------------------------------------------------------

package *targets:
    uv run scripts/package.py {{targets}}

release: build-all
    uv run scripts/release.py

# --- clean ------------------------------------------------------------------

# Drops build artifacts and the fetched SDK but keeps .cargo/, so the next build re-downloads
# no crates.
clean:
    rm -rf target dist sdk

# Also drops the crate registry, for reproducing a genuinely cold build.
clean-all: clean
    rm -rf .cargo
