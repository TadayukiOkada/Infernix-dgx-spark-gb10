#pragma once
// Only the TMA L2-promotion enum is named by the NVFP4 schedule header; the driver API is not used on device.
enum CUtensorMapL2promotion {
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B = 1,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 2,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 3,
};
