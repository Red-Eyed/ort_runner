#pragma once

#include <string>
#include <unordered_map>

#include "npy.hpp"

namespace ort_runner {

// Reads a .npz archive -- a zip of .npy members, as written by numpy.savez -- into a map from
// array name to parsed array. The map key is the member name with its trailing ".npy" stripped,
// so it matches the keyword numpy.savez was called with (and thus a model input name).
//
// Throws std::runtime_error if the file cannot be read, is not a valid zip, or contains a
// compressed member. numpy.savez writes STORED (uncompressed) members; numpy.savez_compressed
// (DEFLATE) is intentionally unsupported -- the error message says to re-save with numpy.savez.
std::unordered_map<std::string, NpyArray> LoadNpz(const std::string &path);

}  // namespace ort_runner
