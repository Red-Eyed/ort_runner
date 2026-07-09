// doctest's main is defined in test_input_synth.cpp; this TU only contributes test cases.
#include <doctest.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy.hpp"

using ort_runner::NpyDType;
using ort_runner::NpyDTypeName;
using ort_runner::NpyDTypeSize;
using ort_runner::ParseNpy;

namespace {

// Builds an in-memory .npy v1.0 image: magic + version + uint16 header length + the header dict
// + raw data. Enough to exercise ParseNpy without touching the filesystem.
std::string MakeNpy(const std::string &descr, bool fortran, const std::string &shape,
                    const std::string &data) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': " +
                         (fortran ? "True" : "False") + ", 'shape': (" + shape + "), }";
    std::string out;
    out += '\x93';
    out += "NUMPY";
    out += '\x01';  // major version 1
    out += '\x00';  // minor version 0
    auto len = static_cast<uint16_t>(header.size());
    out += static_cast<char>(len & 0xff);
    out += static_cast<char>((len >> 8) & 0xff);
    out += header;
    out += data;
    return out;
}

ort_runner::NpyArray Parse(const std::string &image) {
    return ParseNpy(reinterpret_cast<const std::byte *>(image.data()), image.size());
}

std::string FloatBytes(std::initializer_list<float> values) {
    std::string out;
    for (float v : values) {
        char buf[sizeof(float)];
        std::memcpy(buf, &v, sizeof(float));
        out.append(buf, sizeof(float));
    }
    return out;
}

}  // namespace

TEST_CASE("parses a 2x3 float32 array's dtype, shape, and bytes") {
    std::string data = FloatBytes({1, 2, 3, 4, 5, 6});
    auto array = Parse(MakeNpy("<f4", false, "2, 3", data));

    CHECK(array.dtype == NpyDType::f4);
    CHECK(array.shape == std::vector<int64_t>{2, 3});
    REQUIRE(array.data.size() == data.size());
    CHECK(std::memcmp(array.data.data(), data.data(), data.size()) == 0);
}

TEST_CASE("a scalar shape () parses to an empty shape") {
    auto array = Parse(MakeNpy("<f4", false, "", FloatBytes({42})));
    CHECK(array.shape.empty());
    CHECK(array.data.size() == sizeof(float));
}

TEST_CASE("a 1-D shape (3,) parses to a single-axis shape") {
    auto array = Parse(MakeNpy("<i8", false, "3,", std::string(3 * 8, '\0')));
    CHECK(array.dtype == NpyDType::i8);
    CHECK(array.shape == std::vector<int64_t>{3});
}

TEST_CASE("single-byte dtypes carry the '|' byte-order and still parse") {
    CHECK(Parse(MakeNpy("|u1", false, "2,", std::string("\x01\x02", 2))).dtype == NpyDType::u1);
    CHECK(Parse(MakeNpy("|b1", false, "2,", std::string("\x00\x01", 2))).dtype == NpyDType::b1);
    CHECK(Parse(MakeNpy("|i1", false, "1,", std::string("\x05", 1))).dtype == NpyDType::i1);
}

TEST_CASE("bad magic is rejected") {
    std::string junk = "not a npy file at all........";
    CHECK_THROWS_AS(Parse(junk), std::runtime_error);
}

TEST_CASE("Fortran-ordered data is rejected") {
    CHECK_THROWS_AS(Parse(MakeNpy("<f4", true, "2, 3", std::string(24, '\0'))),
                    std::runtime_error);
}

TEST_CASE("big-endian data is rejected") {
    CHECK_THROWS_AS(Parse(MakeNpy(">f4", false, "1,", std::string(4, '\0'))), std::runtime_error);
}

TEST_CASE("an unsupported dtype is rejected") {
    // uint32 is a valid numpy dtype but outside ort_runner's supported subset.
    CHECK_THROWS_AS(Parse(MakeNpy("<u4", false, "1,", std::string(4, '\0'))), std::runtime_error);
}

TEST_CASE("truncated data (fewer bytes than the shape implies) is rejected") {
    CHECK_THROWS_AS(Parse(MakeNpy("<f4", false, "2, 3", FloatBytes({1, 2}))), std::runtime_error);
}

TEST_CASE("dtype size and name cover the supported subset") {
    CHECK(NpyDTypeSize(NpyDType::f4) == 4);
    CHECK(NpyDTypeSize(NpyDType::f8) == 8);
    CHECK(NpyDTypeSize(NpyDType::i8) == 8);
    CHECK(NpyDTypeSize(NpyDType::b1) == 1);
    CHECK(NpyDTypeName(NpyDType::f4) == "float32");
    CHECK(NpyDTypeName(NpyDType::i8) == "int64");
    CHECK(NpyDTypeName(NpyDType::b1) == "bool");
}
