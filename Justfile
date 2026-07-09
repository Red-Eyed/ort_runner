default:
    @just --list

# Build the Linux x86_64 toolchain image
image-linux-x64:
    scripts/build_image.py linux-x64

# Build the Linux aarch64 toolchain image (old-glibc baseline; e.g. Raspberry Pi 4)
image-linux-aarch64:
    scripts/build_image.py linux-aarch64

# Build the Android NDK toolchain image (shared by both Android ABIs)
image-android:
    scripts/build_image.py android-arm64

# Fetch ONNX Runtime (cached on the host, see .gitignore) + configure + build a binary, then
# smoke-test the result. Each recipe builds into its own build-<target>/ directory.
build-linux-x64: image-linux-x64
    scripts/build.py linux-x64
    scripts/smoke.py linux-x64

build-linux-aarch64: image-linux-aarch64
    scripts/build.py linux-aarch64
    scripts/smoke.py linux-aarch64

build-android-arm64: image-android
    scripts/build.py android-arm64
    scripts/smoke.py android-arm64

build-android-armv7: image-android
    scripts/build.py android-armv7
    scripts/smoke.py android-armv7

# Run a Linux build's unit tests
test-linux-x64: build-linux-x64
    scripts/test_linux.py linux-x64

test-linux-aarch64: build-linux-aarch64
    scripts/test_linux.py linux-aarch64

# Run a Linux build against a model (aarch64 runs under emulation on a non-arm64 host)
run-linux-x64 model *args: build-linux-x64
    scripts/run_linux.py linux-x64 {{model}} {{args}}

run-linux-aarch64 model *args: build-linux-aarch64
    scripts/run_linux.py linux-aarch64 {{model}} {{args}}

# Push an Android build + libonnxruntime.so to a connected device/emulator and run it there
run-android-arm64 model *args: build-android-arm64
    scripts/run_android.py android-arm64 {{model}} {{args}}

run-android-armv7 model *args: build-android-armv7
    scripts/run_android.py android-armv7 {{model}} {{args}}

# Package one or more built targets into dist/ort_runner-<version>-<target>.zip
package *targets:
    scripts/package.py {{targets}}

# Build every target, then publish a GitHub release (tag v<version>) with the per-target zips
release: build-linux-x64 build-linux-aarch64 build-android-arm64 build-android-armv7
    scripts/release.py

# Remove build outputs, packaged zips, and the fetched ONNX Runtime SDK
clean:
    rm -rf build-linux-x64 build-linux-aarch64 build-android-arm64 build-android-armv7 dist sdk
