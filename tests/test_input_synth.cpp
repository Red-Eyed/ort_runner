#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "input_synth.hpp"

using ort_runner::DimOverrides;
using ort_runner::ResolveShape;

TEST_CASE("static dims are left untouched") {
    std::vector<int64_t> declared = {1, 3, 224, 224};
    CHECK(ResolveShape(declared, {}, {}, 8) == declared);
}

TEST_CASE("dynamic dims (represented as -1) are substituted with default_dim") {
    std::vector<int64_t> declared = {-1, 3, 224, 224};
    std::vector<int64_t> expected = {8, 3, 224, 224};
    CHECK(ResolveShape(declared, {}, {}, 8) == expected);
}

TEST_CASE("zero-valued dims are treated as dynamic too") {
    std::vector<int64_t> declared = {0, 3};
    std::vector<int64_t> expected = {5, 3};
    CHECK(ResolveShape(declared, {}, {}, 5) == expected);
}

TEST_CASE("a fully dynamic shape resolves every dim") {
    std::vector<int64_t> declared = {-1, -1, -1};
    std::vector<int64_t> expected = {2, 2, 2};
    CHECK(ResolveShape(declared, {}, {}, 2) == expected);
}

TEST_CASE("an empty shape (scalar tensor) resolves to empty") {
    std::vector<int64_t> declared = {};
    CHECK(ResolveShape(declared, {}, {}, 4).empty());
}

TEST_CASE("a named symbolic dim uses its matching --dim override") {
    std::vector<int64_t> declared = {-1, 3, 224, 224};
    std::vector<std::string> symbolic_dims = {"batch", "", "", ""};
    DimOverrides overrides = {{"batch", 16}};
    std::vector<int64_t> expected = {16, 3, 224, 224};
    CHECK(ResolveShape(declared, symbolic_dims, overrides, 8) == expected);
}

TEST_CASE("an override for a name that doesn't appear on this axis is ignored for it") {
    std::vector<int64_t> declared = {-1, 3};
    std::vector<std::string> symbolic_dims = {"batch", ""};
    DimOverrides overrides = {{"seq_len", 99}};  // doesn't match "batch"
    std::vector<int64_t> expected = {8, 3};       // falls back to default_dim
    CHECK(ResolveShape(declared, symbolic_dims, overrides, 8) == expected);
}

TEST_CASE("an anonymous dynamic dim falls back to default_dim regardless of unrelated overrides") {
    std::vector<int64_t> declared = {-1};
    std::vector<std::string> symbolic_dims = {""};  // no symbolic name at all
    DimOverrides overrides = {{"batch", 16}};
    std::vector<int64_t> expected = {8};
    CHECK(ResolveShape(declared, symbolic_dims, overrides, 8) == expected);
}

TEST_CASE("two axes sharing the same symbolic name both get the override") {
    std::vector<int64_t> declared = {-1, -1};
    std::vector<std::string> symbolic_dims = {"N", "N"};
    DimOverrides overrides = {{"N", 5}};
    std::vector<int64_t> expected = {5, 5};
    CHECK(ResolveShape(declared, symbolic_dims, overrides, 1) == expected);
}
