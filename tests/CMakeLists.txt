add_executable(ort_runner_tests
    test_input_synth.cpp
    test_npy.cpp
    ${CMAKE_SOURCE_DIR}/src/input_synth.cpp
    ${CMAKE_SOURCE_DIR}/src/npy.cpp
    ${CMAKE_SOURCE_DIR}/src/npz.cpp
)

target_include_directories(ort_runner_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/doctest
)

target_link_libraries(ort_runner_tests PRIVATE onnxruntime::onnxruntime)

ort_runner_bundle_onnxruntime(ort_runner_tests)

add_test(NAME ort_runner_tests COMMAND ort_runner_tests)
