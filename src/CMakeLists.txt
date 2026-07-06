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

# Keep build-*/bin/ self-contained (directly adb-pushable or runnable in place) by copying the
# resolved onnxruntime shared library next to the binary.
add_custom_command(TARGET ort_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_PROPERTY:onnxruntime::onnxruntime,IMPORTED_LOCATION>"
        "$<TARGET_FILE_DIR:ort_runner>"
)
