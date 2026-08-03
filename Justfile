# Running a released ort_runner. These recipes need Python 3.9+ and, for a device, adb -- no
# Podman, no Rust, no NDK. Developer recipes live under `just dev ...`.

mod dev

default:
    @just --list

# Download released binaries. Defaults to every target; name targets to narrow it.
download-prebuilt *targets:
    uv run scripts/download_prebuilt.py {{ targets }}

# Download and run a released binary for one target.
run target model *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run scripts/download_prebuilt.py {{ target }}
    case "{{ target }}" in
      linux-*) uv run scripts/run_linux.py --source prebuilt {{ target }} {{ model }} {{ args }} ;;
      android-*) uv run scripts/run_android.py --source prebuilt {{ target }} {{ model }} {{ args }} ;;
      *) echo "error: invalid target '{{ target }}'" >&2; exit 2 ;;
    esac

# Unit tests.
test:
    @just dev test

# Everything CI would check.
check:
    @just dev check

# Compatibility aliases kept out of `just --list`.

[private]
run-linux-x64 model *args:
    @just run linux-x64 {{ model }} {{ args }}

[private]
run-linux-aarch64 model *args:
    @just run linux-aarch64 {{ model }} {{ args }}

[private]
run-android-arm64 model *args:
    @just run android-arm64 {{ model }} {{ args }}

[private]
run-android-armv7 model *args:
    @just run android-armv7 {{ model }} {{ args }}

[private]
run-android-x86_64 model *args:
    @just run android-x86_64 {{ model }} {{ args }}

[private]
run-dev-linux-x64 model *args:
    @just dev run linux-x64 {{ model }} {{ args }}

[private]
run-dev-linux-aarch64 model *args:
    @just dev run linux-aarch64 {{ model }} {{ args }}

[private]
run-dev-android-arm64 model *args:
    @just dev run android-arm64 {{ model }} {{ args }}

[private]
run-dev-android-armv7 model *args:
    @just dev run android-armv7 {{ model }} {{ args }}

[private]
run-dev-android-x86_64 model *args:
    @just dev run android-x86_64 {{ model }} {{ args }}

[private]
build-linux-x64:
    @just dev build linux-x64

[private]
build-linux-aarch64:
    @just dev build linux-aarch64

[private]
build-android-arm64:
    @just dev build android-arm64

[private]
build-android-armv7:
    @just dev build android-armv7

[private]
build-android-x86_64:
    @just dev build android-x86_64

[private]
image-linux-x64:
    @just dev image-linux-x64

[private]
image-linux-aarch64:
    @just dev image-linux-aarch64

[private]
image-android:
    @just dev image-android

[private]
images:
    @just dev images

[private]
build-all:
    @just dev build-all

[private]
fetch:
    @just dev fetch

[private]
lint:
    @just dev lint

[private]
fmt:
    @just dev fmt

[private]
test-e2e:
    @just dev test-e2e

[private]
package *targets:
    @just dev package {{ targets }}

[private]
release:
    @just dev release

[private]
clean:
    @just dev clean

[private]
clean-all:
    @just dev clean-all
