default:
    @just --list

# Build the native-Linux toolchain image. Platform is explicit (derived from the host arch)
# rather than left to podman's default resolution -- observed podman pick up a stale
# linux/amd64 layer here after the linux/amd64 android image pulled the same debian:12-slim
# base, silently producing an emulated (not native) image on an arm64 host.
image-linux:
    podman build --platform "linux/$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/')" \
        -t ort-runner-builder-linux -f podman/Containerfile.linux podman/

# Build the Android NDK toolchain image. Always targets linux/amd64: the NDK only ships a
# Linux host toolchain for x86_64, regardless of the arm64-v8a Android target ABI, so this
# also has to run under QEMU emulation on an arm64 host/Podman VM (e.g. Apple Silicon).
image-android:
    podman build --platform linux/amd64 -t ort-runner-builder-android -f podman/Containerfile.android podman/

# Fetch ONNX Runtime (cached on the host, see .gitignore) + configure + build the native
# Linux binary into build-linux/
build-linux: image-linux
    podman run --rm -v "{{justfile_directory()}}":/workspace:Z -w /workspace ort-runner-builder-linux \
        bash -c "scripts/fetch_onnxruntime.sh linux && cmake --preset linux -B build-linux && cmake --build build-linux -j"

# Fetch ONNX Runtime + configure + cross-compile the Android arm64-v8a binary into build-android/
build-android: image-android
    podman run --rm --platform linux/amd64 -v "{{justfile_directory()}}":/workspace:Z -w /workspace ort-runner-builder-android \
        bash -c "scripts/fetch_onnxruntime.sh android && cmake --preset android-arm64 -B build-android && cmake --build build-android -j"

# Run the native Linux build's unit tests
test-linux: build-linux
    build-linux/tests/ort_runner_tests

# Run the native Linux build locally against a model
run-linux model *args: build-linux
    LD_LIBRARY_PATH=build-linux/bin build-linux/bin/ort_runner --model {{model}} {{args}}

# Push the Android build + libonnxruntime.so to a connected device/emulator and run it there
push-android model *args: build-android
    adb push build-android/bin/ort_runner /data/local/tmp/ort_runner
    adb push build-android/bin/libonnxruntime.so /data/local/tmp/libonnxruntime.so
    adb shell chmod +x /data/local/tmp/ort_runner
    adb shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/ort_runner --model {{model}} {{args}}"

# Remove build outputs and the fetched ONNX Runtime SDK
clean:
    rm -rf build-linux build-android sdk
