default:
    @just --list

# Build the native-Linux toolchain image
image-linux:
    scripts/build_image.py linux

# Build the Android NDK toolchain image
image-android:
    scripts/build_image.py android

# Fetch ONNX Runtime (cached on the host, see .gitignore) + configure + build the native
# Linux binary into build-linux/
build-linux: image-linux
    scripts/build.py linux

# Fetch ONNX Runtime + configure + cross-compile the Android arm64-v8a binary into build-android/
build-android: image-android
    scripts/build.py android

# Run the native Linux build's unit tests
test-linux: build-linux
    build-linux/tests/ort_runner_tests

# Run the native Linux build locally against a model
run-linux model *args: build-linux
    scripts/run_linux.py {{model}} {{args}}

# Push the Android build + libonnxruntime.so to a connected device/emulator and run it there
run-android model *args: build-android
    scripts/run_android.py {{model}} {{args}}

# Remove build outputs and the fetched ONNX Runtime SDK
clean:
    rm -rf build-linux build-android sdk
