#!/usr/bin/env bash
# Downloads a pinned official ONNX Runtime distribution and unpacks it into sdk/, printing
# the resulting ORT_RUNNER_SDK_DIR value for the caller (CMakePresets.json / Justfile) to use.
set -euo pipefail

ORT_VERSION="1.27.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
SDK_DIR="${REPO_ROOT}/sdk"

usage() {
    echo "Usage: $0 <linux|android>" >&2
    exit 1
}

[[ $# -eq 1 ]] || usage
target="$1"

mkdir -p "${SDK_DIR}"

case "${target}" in
    linux)
        arch="$(uname -m)"
        case "${arch}" in
            x86_64) asset="onnxruntime-linux-x64-${ORT_VERSION}.tgz" ;;
            aarch64) asset="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz" ;;
            *)
                echo "Unsupported host architecture for a linux ONNX Runtime build: ${arch}" >&2
                exit 1
                ;;
        esac
        # Fixed name regardless of detected arch, so CMakePresets.json's "linux" preset works
        # unmodified whether the build runs on an x86_64 or aarch64 host/container.
        dest_dir="${SDK_DIR}/onnxruntime-linux"
        if [[ -f "${dest_dir}/include/onnxruntime_cxx_api.h" ]]; then
            echo "Already present: ${dest_dir}" >&2
        else
            url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${asset}"
            echo "Downloading ${url}" >&2
            curl -fL "${url}" -o "${SDK_DIR}/${asset}"
            tar -xzf "${SDK_DIR}/${asset}" -C "${SDK_DIR}"
            rm -f "${SDK_DIR}/${asset}"
        fi
        echo "ORT_RUNNER_SDK_DIR=${dest_dir}"
        ;;
    android)
        aar="onnxruntime-android-${ORT_VERSION}.aar"
        dest_dir="${SDK_DIR}/onnxruntime-android-arm64"
        if [[ -f "${dest_dir}/headers/onnxruntime_cxx_api.h" ]]; then
            echo "Already present: ${dest_dir}" >&2
        else
            url="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/${aar}"
            echo "Downloading ${url}" >&2
            curl -fL "${url}" -o "${SDK_DIR}/${aar}"
            mkdir -p "${dest_dir}"
            # The AAR is a zip; only headers/ and the arm64-v8a shared lib are needed (the
            # Java/JNI glue lib and classes.jar are irrelevant to a native CLI).
            unzip -o "${SDK_DIR}/${aar}" "headers/*" "jni/arm64-v8a/libonnxruntime.so" -d "${dest_dir}"
            rm -f "${SDK_DIR}/${aar}"
        fi
        echo "ORT_RUNNER_SDK_DIR=${dest_dir}"
        ;;
    *)
        usage
        ;;
esac
