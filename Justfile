# Running a released ort_runner. These recipes need Python 3.9+ and, for a device, adb -- no
# Podman, no Rust, no NDK. The download is idempotent, so the first run fetches and later ones do
# not.
#
# Building from source is a developer's cost, and those recipes live in dev.just: `build-*`,
# `test`, `lint`, `check`, `run-dev-*`, `package`, `release`, `clean`. They are imported below
# rather than namespaced, so they are invoked exactly as before -- `just check`, not
# `just dev::check` -- and `just --list` shows everything in one list.

import 'dev.just'

default:
    @just --list

# --- run --------------------------------------------------------------------

# Download released binaries. Defaults to every target; name targets to narrow it.
download-prebuilt *targets:
    uv run scripts/download_prebuilt.py {{targets}}

run-linux-x64 model *args:
    uv run scripts/download_prebuilt.py linux-x64
    uv run scripts/run_linux.py --source prebuilt linux-x64 {{model}} {{args}}

run-linux-aarch64 model *args:
    uv run scripts/download_prebuilt.py linux-aarch64
    uv run scripts/run_linux.py --source prebuilt linux-aarch64 {{model}} {{args}}

# Pushes the binary + libonnxruntime.so to a connected device and runs it there. Needs only adb.
run-android-arm64 model *args:
    uv run scripts/download_prebuilt.py android-arm64
    uv run scripts/run_android.py --source prebuilt android-arm64 {{model}} {{args}}

run-android-armv7 model *args:
    uv run scripts/download_prebuilt.py android-armv7
    uv run scripts/run_android.py --source prebuilt android-armv7 {{model}} {{args}}

# The emulator ABI.
run-android-x86_64 model *args:
    uv run scripts/download_prebuilt.py android-x86_64
    uv run scripts/run_android.py --source prebuilt android-x86_64 {{model}} {{args}}
