cmake_minimum_required(VERSION 3.21)
project(ort_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(ORT_RUNNER_SDK_DIR "" CACHE PATH
    "Directory populated by scripts/fetch_onnxruntime.sh (e.g. sdk/onnxruntime-linux-x64)")
if(ORT_RUNNER_SDK_DIR STREQUAL "")
    message(FATAL_ERROR
        "ORT_RUNNER_SDK_DIR is not set. Run scripts/fetch_onnxruntime.sh, then configure with "
        "-DORT_RUNNER_SDK_DIR=<path-it-printed>, or use one of the CMakePresets.json presets.")
endif()

include(cmake/FindOnnxRuntime.cmake)

add_library(argparse INTERFACE)
target_include_directories(argparse INTERFACE third_party/argparse)

add_library(nanobench INTERFACE)
target_include_directories(nanobench INTERFACE third_party/nanobench)

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE third_party/json)

add_library(magic_enum INTERFACE)
target_include_directories(magic_enum INTERFACE third_party/magic_enum)

add_subdirectory(src)

if(NOT CMAKE_CROSSCOMPILING)
    enable_testing()
    add_subdirectory(tests)
endif()
