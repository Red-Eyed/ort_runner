configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/version.hpp.in
    ${CMAKE_BINARY_DIR}/generated/version.hpp
    @ONLY
)

add_executable(ort_runner
    main.cpp
    config.cpp
    ort_session.cpp
    input_synth.cpp
    npy.cpp
    npz.cpp
    input_load.cpp
    benchmark.cpp
    report.cpp
)

target_include_directories(ort_runner PRIVATE ${CMAKE_BINARY_DIR}/generated)

target_link_libraries(ort_runner PRIVATE
    onnxruntime::onnxruntime
    argparse
    nanobench
    nlohmann_json
    magic_enum
)

set_target_properties(ort_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

ort_runner_bundle_onnxruntime(ort_runner)
