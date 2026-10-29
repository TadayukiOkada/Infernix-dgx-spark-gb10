#pragma once

namespace infernix::ops::detail {

// Number of streaming multiprocessors on the current device, queried once and cached for the
// process lifetime. The whole build targets GB10 (sm_121a, 48 SMs); launch geometry is tuned
// directly for this part, so this value is only consulted by the split-K wave planner to size its
// final resident wave. Any query failure degrades to the GB10 reference count rather than aborting
// a launch.
int device_sm_count();

} // namespace infernix::ops::detail
