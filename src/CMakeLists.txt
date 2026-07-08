add_executable(ort_runner
    main.cpp
    config.cpp
    ort_session.cpp
    input_synth.cpp
    benchmark.cpp
    report.cpp
)

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
