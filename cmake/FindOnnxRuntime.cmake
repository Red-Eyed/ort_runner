# Locates a prebuilt ONNX Runtime distribution under ORT_RUNNER_SDK_DIR and defines the
# imported target onnxruntime::onnxruntime. Upstream ships a lib/cmake/onnxruntime config
# that is mismatched/broken across releases (see onnxruntime issues #24003, #26186), so this
# looks at the directory layout directly instead of using find_package(onnxruntime CONFIG).

if(EXISTS "${ORT_RUNNER_SDK_DIR}/include/onnxruntime_cxx_api.h")
    # Official onnxruntime-linux-*.tgz release layout: flat include/ + lib/libonnxruntime.so,
    # itself a dev symlink to lib/libonnxruntime.so.1 (the SONAME actually embedded in the
    # library and what ort_runner's NEEDED entry records). Point at the .so.1 name directly so
    # that whatever copies IMPORTED_LOCATION next to the binary (see
    # ort_runner_bundle_onnxruntime below) produces a file named the way the dynamic linker
    # will actually look for it, instead of the unversioned dev-symlink name.
    set(_ort_runner_include_dir "${ORT_RUNNER_SDK_DIR}/include")
    set(_ort_runner_lib "${ORT_RUNNER_SDK_DIR}/lib/libonnxruntime.so.1")
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

# Makes `target` locate onnxruntime's shared library next to itself at runtime: copies the
# library (under its real SONAME-based filename, via IMPORTED_LOCATION above) beside the
# target's output, and points its RPATH at "$ORIGIN" instead of the SDK's build-tree location.
# That keeps build-*/bin self-contained -- directly adb-pushable or runnable in place from any
# directory -- without needing LD_LIBRARY_PATH or the sdk/ directory around.
function(ort_runner_bundle_onnxruntime target)
    set_target_properties(${target} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_PROPERTY:onnxruntime::onnxruntime,IMPORTED_LOCATION>"
            "$<TARGET_FILE_DIR:${target}>")
endfunction()
