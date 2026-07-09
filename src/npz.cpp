#include "npz.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace ort_runner {

namespace {

// Little-endian field readers over the in-memory archive. Every access is bounds-checked so a
// truncated or crafted zip yields a clean std::runtime_error rather than an out-of-bounds read.
constexpr uint32_t kEocdSignature = 0x06054b50;
constexpr uint32_t kCentralDirSignature = 0x02014b50;
constexpr size_t kEocdMinSize = 22;
constexpr size_t kCentralDirFixedSize = 46;
constexpr size_t kLocalHeaderFixedSize = 30;

uint32_t ReadU32(const std::string &buf, size_t offset) {
    if (offset + 4 > buf.size()) throw std::runtime_error("corrupt .npz: read past end of file");
    const auto *p = reinterpret_cast<const unsigned char *>(buf.data() + offset);
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t ReadU16(const std::string &buf, size_t offset) {
    if (offset + 2 > buf.size()) throw std::runtime_error("corrupt .npz: read past end of file");
    const auto *p = reinterpret_cast<const unsigned char *>(buf.data() + offset);
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

// Scans backward from the end for the End Of Central Directory signature (the record is at the
// very end of a zip, after an optional comment of up to 65535 bytes).
size_t FindEocd(const std::string &buf) {
    if (buf.size() < kEocdMinSize) throw std::runtime_error("corrupt .npz: file too small to be a zip");
    for (size_t offset = buf.size() - kEocdMinSize + 1; offset-- > 0;) {
        if (ReadU32(buf, offset) == kEocdSignature) return offset;
    }
    throw std::runtime_error("corrupt .npz: end-of-central-directory record not found");
}

std::string ReadFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open .npz file: " + path);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// One central-directory entry, reduced to the fields we need to locate and validate its data.
struct Member {
    std::string name;
    uint16_t compression_method;
    uint32_t uncompressed_size;
    uint32_t local_header_offset;
};

Member ReadCentralDirEntry(const std::string &buf, size_t offset, size_t &next_offset) {
    if (ReadU32(buf, offset) != kCentralDirSignature) {
        throw std::runtime_error("corrupt .npz: bad central-directory entry signature");
    }
    Member member;
    member.compression_method = ReadU16(buf, offset + 10);
    member.uncompressed_size = ReadU32(buf, offset + 24);
    uint16_t name_len = ReadU16(buf, offset + 28);
    uint16_t extra_len = ReadU16(buf, offset + 30);
    uint16_t comment_len = ReadU16(buf, offset + 32);
    member.local_header_offset = ReadU32(buf, offset + 42);

    size_t name_offset = offset + kCentralDirFixedSize;
    if (name_offset + name_len > buf.size()) throw std::runtime_error("corrupt .npz: bad member name length");
    member.name = buf.substr(name_offset, name_len);

    next_offset = name_offset + name_len + extra_len + comment_len;
    return member;
}

// The member's raw bytes begin after its *local* header, whose variable-length name/extra fields
// can differ from the central directory's, so we re-read their lengths here.
std::string ExtractMemberData(const std::string &buf, const Member &member) {
    if (member.compression_method != 0) {
        throw std::runtime_error(
            "compressed .npz member '" + member.name +
            "' is not supported; re-save with numpy.savez (not numpy.savez_compressed)");
    }
    if (member.uncompressed_size == 0xFFFFFFFF || member.local_header_offset == 0xFFFFFFFF) {
        throw std::runtime_error("ZIP64 .npz archives are not supported");
    }

    size_t local = member.local_header_offset;
    uint16_t name_len = ReadU16(buf, local + 26);
    uint16_t extra_len = ReadU16(buf, local + 28);
    size_t data_offset = local + kLocalHeaderFixedSize + name_len + extra_len;
    if (data_offset + member.uncompressed_size > buf.size()) {
        throw std::runtime_error("corrupt .npz: member data extends past end of file");
    }
    return buf.substr(data_offset, member.uncompressed_size);
}

// numpy stores each array under "<key>.npy"; strip that suffix to recover the savez keyword.
std::string StripNpySuffix(const std::string &name) {
    constexpr std::string_view suffix = ".npy";
    if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return name.substr(0, name.size() - suffix.size());
    }
    return name;
}

}  // namespace

std::unordered_map<std::string, NpyArray> LoadNpz(const std::string &path) {
    std::string buf = ReadFile(path);

    size_t eocd = FindEocd(buf);
    uint16_t entry_count = ReadU16(buf, eocd + 10);
    uint32_t cd_offset = ReadU32(buf, eocd + 16);

    std::unordered_map<std::string, NpyArray> arrays;
    size_t offset = cd_offset;
    for (uint16_t i = 0; i < entry_count; ++i) {
        size_t next_offset = 0;
        Member member = ReadCentralDirEntry(buf, offset, next_offset);
        std::string data = ExtractMemberData(buf, member);
        arrays.emplace(StripNpySuffix(member.name),
                       ParseNpy(reinterpret_cast<const std::byte *>(data.data()), data.size()));
        offset = next_offset;
    }
    return arrays;
}

}  // namespace ort_runner
