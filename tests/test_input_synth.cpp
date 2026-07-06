#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "input_synth.hpp"

using ort_runner::ResolveShape;

TEST_CASE("static dims are left untouched") {
    std::vector<int64_t> declared = {1, 3, 224, 224};
    CHECK(ResolveShape(declared, 8) == declared);
}

TEST_CASE("dynamic dims (represented as -1) are substituted with default_dim") {
    std::vector<int64_t> declared = {-1, 3, 224, 224};
    std::vector<int64_t> expected = {8, 3, 224, 224};
    CHECK(ResolveShape(declared, 8) == expected);
}

TEST_CASE("zero-valued dims are treated as dynamic too") {
    std::vector<int64_t> declared = {0, 3};
    std::vector<int64_t> expected = {5, 3};
    CHECK(ResolveShape(declared, 5) == expected);
}

TEST_CASE("a fully dynamic shape resolves every dim") {
    std::vector<int64_t> declared = {-1, -1, -1};
    std::vector<int64_t> expected = {2, 2, 2};
    CHECK(ResolveShape(declared, 2) == expected);
}

TEST_CASE("an empty shape (scalar tensor) resolves to empty") {
    std::vector<int64_t> declared = {};
    CHECK(ResolveShape(declared, 4).empty());
}
