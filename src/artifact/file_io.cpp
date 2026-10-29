#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace infernix::artifact {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

} // namespace


InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        // A filesystem without direct I/O (tmpfs, 9p, some FUSE mounts and Docker bind mounts)
        // refuses O_DIRECT: the same aligned reads then go through a buffered descriptor.
        if (direct_fd_ < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
            direct_fd_ = ::fcntl(fd_, F_DUPFD_CLOEXEC, 0);
        }
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    // One pread may return less than requested before the end of the file (FUSE and network
    // filesystems cap a read at their own maximum), so read until the range is full or the file
    // ends, as the Windows path does (upstream Neroued/ninfer#381). O_DIRECT cannot continue from
    // an unaligned position, so after such a short read the buffered descriptor finishes the range.
    std::size_t total = 0;
    while (total < destination.size()) {
        const int fd    = total % kPayloadAlignment ? fd_ : direct_fd_;
        const auto rest = destination.subspan(total);
        const auto read = ::pread(fd, rest.data(), rest.size(), file_offset(offset + total));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "direct pread");
        }
        if (!read) { break; }
        total += static_cast<std::size_t>(read);
    }
    return total;
}

} // namespace infernix::artifact
