# NVRTC host-header shim

Kernel Tuner's `nvcuda` backend compiles with NVRTC, which has no C++ standard library and
cannot parse the host half of the product headers. This directory is placed *before* `src/` on
the include path and replaces only host-only headers (`core/*`, the Operands structs, `<stdexcept>`)
with empty or minimal stand-ins. Every `__device__`/`__global__` definition under test is still
the unmodified file from `src/`. Nothing here may define device code.
