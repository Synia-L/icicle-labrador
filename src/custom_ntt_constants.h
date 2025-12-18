#pragma once

#include <cstdint>

constexpr int kCustomPolyDegree = 64;
constexpr int kCustomDomainSize = 128;
constexpr int kCustomLimbs = 2;
constexpr uint64_t kCustomModulusQ = 0x3b880000f7000001ULL;

#ifndef __CUDACC__
using cudaError_t = int;
#endif

struct CustomNttConstants {
    uint32_t inv_twiddles_bb[kCustomDomainSize];
    uint32_t inv_twiddles_kb[kCustomDomainSize];
    uint32_t coset_bb[kCustomPolyDegree];
    uint32_t coset_kb[kCustomPolyDegree];
    uint32_t inv_coset_bb[kCustomPolyDegree];
    uint32_t inv_coset_kb[kCustomPolyDegree];
    uint32_t inv_N_lo;
    uint32_t inv_N_hi;
};

#ifdef __cplusplus
extern "C" {
#endif
cudaError_t custom_ntt_get_constants(CustomNttConstants* host_consts);
#ifdef __cplusplus
}
#endif

