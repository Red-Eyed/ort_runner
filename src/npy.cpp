#include "npy.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ort_runner {

namespace {

// The 6-byte magic prefix every .npy file starts with: \x93 N U M P Y.
constexpr std::array<unsigned char, 6> kMagic = {0x93, 'N', 'U', 'M', 'P', 'Y'};

uint32_t ReadLE(const std::byte *p, size_t n) {
    uint32_t value = 0;
    for (size_t i = 0; i < n; ++i) {
        value |= static_cast<uint32_t>(static_cast<unsigned char>(p[i])) << (8 * i);
    }
    return value;
}

// Locates the ASCII header dict and where the raw element data begins. The header length field
// is 2 bytes in format v1.x and 4 bytes in v2.x+; both start at byte 8.
struct HeaderLayout {
    std::string header;
    size_t data_offset;
};

HeaderLayout LocateHeader(const std::byte *data, size_t size) {
    if (size < 10 || std::memcmp(data, kMagic.data(), kMagic.size()) != 0) {
        throw std::runtime_error("not a .npy file (bad magic)");
    }
    unsigned char major = static_cast<unsigned char>(data[6]);
    size_t len_bytes = major >= 2 ? 4 : 2;
    size_t len_offset = 8;
    if (size < len_offset + len_bytes) throw std::runtime_error("truncated .npy header length");

    size_t header_len = ReadLE(data + len_offset, len_bytes);
    size_t header_start = len_offset + len_bytes;
    if (size < header_start + header_len) throw std::runtime_error("truncated .npy header");

    return HeaderLayout{
        std::string(reinterpret_cast<const char *>(data + header_start), header_len),
        header_start + header_len,
    };
}

// Returns the value token after `key`'s colon in the Python-dict-literal header, stopping at the
// first delimiter in `stop`. Throws if the key is absent.
std::string FieldAfter(const std::string &header, const std::string &key, const char *stop) {
    size_t key_pos = header.find(key);
    if (key_pos == std::string::npos) throw std::runtime_error("missing '" + key + "' in .npy header");
    size_t colon = header.find(':', key_pos + key.size());
    if (colon == std::string::npos) throw std::runtime_error("malformed '" + key + "' in .npy header");

    size_t start = header.find_first_not_of(" '\t", colon + 1);
    size_t end = header.find_first_of(stop, start);
    return header.substr(start, end - start);
}

NpyDType ParseDescr(const std::string &header) {
    std::string descr = FieldAfter(header, "'descr'", ",'}");
    if (descr.empty()) throw std::runtime_error("empty 'descr' in .npy header");

    char order = descr.front();
    if (order == '>') {
        throw std::runtime_error("big-endian .npy data is not supported (descr '" + descr + "')");
    }
    // '<' little-endian, '=' native (little-endian on all supported targets), '|' not-applicable
    // (single-byte types). Any of these is byte-compatible with a straight memcpy.
    std::string code = (order == '<' || order == '=' || order == '|') ? descr.substr(1) : descr;

    if (code == "f4") return NpyDType::f4;
    if (code == "f8") return NpyDType::f8;
    if (code == "i8") return NpyDType::i8;
    if (code == "i4") return NpyDType::i4;
    if (code == "i2") return NpyDType::i2;
    if (code == "i1") return NpyDType::i1;
    if (code == "u1") return NpyDType::u1;
    if (code == "b1") return NpyDType::b1;
    throw std::runtime_error("unsupported .npy dtype '" + descr +
                             "' (supported: float32/float64/int64/int32/int16/int8/uint8/bool)");
}

std::vector<int64_t> ParseShape(const std::string &header) {
    size_t key = header.find("'shape'");
    if (key == std::string::npos) throw std::runtime_error("missing 'shape' in .npy header");
    size_t open = header.find('(', key);
    size_t close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("malformed 'shape' in .npy header");
    }

    std::vector<int64_t> shape;
    std::string dims = header.substr(open + 1, close - open - 1);
    size_t pos = 0;
    while (pos < dims.size()) {
        size_t next = dims.find(',', pos);
        std::string token = dims.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        size_t first = token.find_first_not_of(" \t");
        if (first != std::string::npos) {
            shape.push_back(std::stoll(token.substr(first)));
        }
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return shape;
}

int64_t NumElements(const std::vector<int64_t> &shape) {
    int64_t count = 1;
    for (int64_t dim : shape) count *= dim;
    return count;
}

}  // namespace

size_t NpyDTypeSize(NpyDType dtype) {
    switch (dtype) {
        case NpyDType::f8:
        case NpyDType::i8: return 8;
        case NpyDType::f4:
        case NpyDType::i4: return 4;
        case NpyDType::i2: return 2;
        case NpyDType::i1:
        case NpyDType::u1:
        case NpyDType::b1: return 1;
    }
    return 0;
}

std::string NpyDTypeName(NpyDType dtype) {
    switch (dtype) {
        case NpyDType::f4: return "float32";
        case NpyDType::f8: return "float64";
        case NpyDType::i8: return "int64";
        case NpyDType::i4: return "int32";
        case NpyDType::i2: return "int16";
        case NpyDType::i1: return "int8";
        case NpyDType::u1: return "uint8";
        case NpyDType::b1: return "bool";
    }
    return "unknown";
}

NpyArray ParseNpy(const std::byte *data, size_t size) {
    HeaderLayout layout = LocateHeader(data, size);

    if (FieldAfter(layout.header, "'fortran_order'", ",'}") != "False") {
        throw std::runtime_error(
            "Fortran-ordered .npy data is not supported; re-save with "
            "numpy.ascontiguousarray(array) before numpy.save");
    }

    NpyArray array;
    array.dtype = ParseDescr(layout.header);
    array.shape = ParseShape(layout.header);

    size_t expected_bytes = static_cast<size_t>(NumElements(array.shape)) * NpyDTypeSize(array.dtype);
    size_t available = size - layout.data_offset;
    if (available < expected_bytes) {
        throw std::runtime_error("truncated .npy data (expected " + std::to_string(expected_bytes) +
                                 " bytes, found " + std::to_string(available) + ")");
    }

    const std::byte *begin = data + layout.data_offset;
    array.data.assign(begin, begin + expected_bytes);
    return array;
}

NpyArray LoadNpy(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open .npy file: " + path);

    std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return ParseNpy(reinterpret_cast<const std::byte *>(bytes.data()), bytes.size());
}

}  // namespace ort_runner
