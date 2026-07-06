# Locates a prebuilt ONNX Runtime distribution under ORT_RUNNER_SDK_DIR and defines the
# imported target onnxruntime::onnxruntime. Upstream ships a lib/cmake/onnxruntime config
# that is mismatched/broken across releases (see onnxruntime issues #24003, #26186), so this
# looks at the directory layout directly instead of using find_package(onnxruntime CONFIG).

if(EXISTS "${ORT_RUNNER_SDK_DIR}/include/onnxruntime_cxx_api.h")
    # Official onnxruntime-linux-*.tgz release layout: flat include/ + lib/libonnxruntime.so
    set(_ort_runner_include_dir "${ORT_RUNNER_SDK_DIR}/include")
    set(_ort_runner_lib "${ORT_RUNNER_SDK_DIR}/lib/libonnxruntime.so")
elseif(EXISTS "${ORT_RUNNER_SDK_DIR}/headers/onnxruntime_cxx_api.h")
    # onnxruntime-android AAR layout, extracted: headers/ + jni/<ANDROID_ABI>/libonnxruntime.so
    if(NOT ANDROID_ABI)
        message(FATAL_ERROR
            "ORT_RUNNER_SDK_DIR looks like an extracted Android AAR, but ANDROID_ABI is not "
            "set. Configure with the Android NDK toolchain file (see CMakePresets.json).")
    endif()
    set(_ort_runner_include_dir "${ORT_RUNNER_SDK_DIR}/headers")
    set(_ort_runner_lib "${ORT_RUNNER_SDK_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so")
else()
    message(FATAL_ERROR
        "Could not find onnxruntime_cxx_api.h under '${ORT_RUNNER_SDK_DIR}/include' or "
        "'${ORT_RUNNER_SDK_DIR}/headers'. Run scripts/fetch_onnxruntime.sh first.")
endif()

if(NOT EXISTS "${_ort_runner_lib}")
    message(FATAL_ERROR "Expected onnxruntime shared library not found at ${_ort_runner_lib}")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${_ort_runner_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${_ort_runner_include_dir}")
