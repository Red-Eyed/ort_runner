#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ort_runner {

// The numpy element types that map onto ort_runner's supported ORT tensor element subset.
// Names mirror numpy's dtype char codes (f4 == float32, i8 == int64, b1 == bool, ...).
enum class NpyDType { f4, f8, i8, i4, i2, i1, u1, b1 };

struct NpyArray {
    NpyDType dtype;
    std::vector<int64_t> shape;    // C-contiguous, row-major
    std::vector<std::byte> data;   // raw little-endian element bytes
};

// Size in bytes of one element of `dtype`.
size_t NpyDTypeSize(NpyDType dtype);

// numpy-style name ("float32", "int64", ...), for error messages.
std::string NpyDTypeName(NpyDType dtype);

// Parses an in-memory .npy v1.0/v2.0 buffer (magic + header dict + raw data). Throws
// std::runtime_error on a malformed header, a Fortran-order array, big-endian data, or a dtype
// outside the supported subset. Pure: no I/O, so it is unit-testable without a file or model.
NpyArray ParseNpy(const std::byte *data, size_t size);

// Reads a .npy file from disk and delegates to ParseNpy. Throws std::runtime_error if the file
// cannot be opened or read.
NpyArray LoadNpy(const std::string &path);

}  // namespace ort_runner
