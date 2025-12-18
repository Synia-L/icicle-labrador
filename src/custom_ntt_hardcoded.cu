#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <stdexcept>

#include "custom_ntt_constants.h"

// ============ 常量定义 ============
const int POLY_DEGREE = kCustomPolyDegree;
const int LIMBS_COUNT = kCustomLimbs;
const int DOMAIN_SIZE = kCustomDomainSize;  // 2 * POLY_DEGREE

const uint64_t MODULUS_Q = kCustomModulusQ;

// ============ 64-bit helpers ============
__host__ __device__ __forceinline__ uint64_t combine_u64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

__host__ __device__ __forceinline__ void split_u64(uint64_t value, uint32_t& lo, uint32_t& hi) {
    lo = static_cast<uint32_t>(value & 0xffffffffULL);
    hi = static_cast<uint32_t>(value >> 32);
}

__device__ __forceinline__ uint64_t load_value(const uint32_t* data, int idx) {
    return combine_u64(data[idx * LIMBS_COUNT], data[idx * LIMBS_COUNT + 1]);
}

__device__ __forceinline__ void store_value(uint32_t* data, int idx, uint64_t value) {
    uint32_t lo, hi;
    split_u64(value, lo, hi);
    data[idx * LIMBS_COUNT] = lo;
    data[idx * LIMBS_COUNT + 1] = hi;
}

__device__ __forceinline__ uint64_t mod_add64(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    if (sum >= MODULUS_Q || sum < a) sum -= MODULUS_Q;
    return sum;
}

__device__ __forceinline__ uint64_t mod_sub64(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (MODULUS_Q - (b - a));
}

__device__ __forceinline__ uint64_t mod_mul64(uint64_t a, uint64_t b) {
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return static_cast<uint64_t>(prod % MODULUS_Q);
}

__device__ __forceinline__ uint64_t load_constant_pair(const uint32_t* lo_arr, const uint32_t* hi_arr, int idx) {
    return combine_u64(lo_arr[idx], hi_arr[idx]);
}

// ============ 设备端常量内存============

// Twiddles: psi^i for i=0..127
__constant__ uint32_t g_device_twiddles_bb[DOMAIN_SIZE] = {
    0x1, 0xd2ff7a61, 0x80335dfe, 0x3cde74dc, 0xb5034d87, 0x11949333, 0x79aeb64e, 0x8d8d9c0a,
    0x5d70079e, 0xc5913f17, 0x25a5398d, 0x3e92aff6, 0x5c14f09f, 0x6cff9bac, 0xf9dd14a6, 0x42e9aa9a,
    0x5f1820c8, 0xbd6f0dcc, 0x54c37af9, 0x89595a89, 0xa5a186b7, 0x6d490d2d, 0xf7133592, 0xed807a9c,
    0x4590fffa, 0x797b601e, 0xf7b78d29, 0xb93b5f7d, 0x3b411a10, 0x33b3184f, 0x1820d679, 0xfc5b8523,
    0x9c24e23a, 0x1a2b3937, 0x5cc1a934, 0xfc9305c0, 0x98472985, 0xcfca8121, 0xe8f26079, 0xb9049dcc,
    0xe26b80d7, 0x57c34fd7, 0x17a4dc2c, 0xcee70043, 0x677ba342, 0x97393de2, 0x6cbe4058, 0x8b7c2dd6,
    0xad9cf002, 0x164dce89, 0xa1d94839, 0x232fa802, 0xe460b959, 0xad73ea1e, 0x913cea47, 0xcdaa57a8,
    0xdf651439, 0x19415948, 0x422ec536, 0x21015b11, 0xf43cbeca, 0x3c491d4c, 0x71d753db, 0xda1ea31c,
    0xf7000000, 0x240085a0, 0x76cca203, 0xba218b25, 0x41fcb27a, 0xe56b6cce, 0x7d5149b3, 0x697263f7,
    0x998ff863, 0x316ec0ea, 0xd15ac674, 0xb86d500b, 0x9aeb0f62, 0x8a006455, 0xfd22eb5b, 0xb4165567,
    0x97e7df39, 0x3990f235, 0xa23c8508, 0x6da6a578, 0x515e794a, 0x89b6f2d4, 0xffecca6f, 0x97f8565,
    0xb16f0007, 0x7d849fe3, 0xff4872d8, 0x3dc4a084, 0xbbbee5f1, 0xc34ce7b2, 0xdedf2988, 0xfaa47ade,
    0x5adb1dc7, 0xdcd4c6ca, 0x9a3e56cd, 0xfa6cfa41, 0x5eb8d67c, 0x27357ee0, 0xe0d9f88, 0x3dfb6235,
    0x14947f2a, 0x9f3cb02a, 0xdf5b23d5, 0x2818ffbe, 0x8f845cbf, 0x5fc6c21f, 0x8a41bfa9, 0x6b83d22b,
    0x49630fff, 0xe0b23178, 0x5526b7c8, 0xd3d057ff, 0x129f46a8, 0x498c15e3, 0x65c315ba, 0x2955a859,
    0x179aebc8, 0xddbea6b9, 0xb4d13acb, 0xd5fea4f0, 0x2c34137, 0xbab6e2b5, 0x8528ac26, 0x1ce15ce5
};

__constant__ uint32_t g_device_twiddles_kb[DOMAIN_SIZE] = {
    0x0, 0x519c7d0, 0x46b9830, 0x38c4d3c3, 0x6c544d, 0x213a516c, 0x2daadd0e, 0x1ca39aeb,
    0x2af38450, 0x21096b19, 0x3934f47e, 0x1c3d3693, 0x19dde0d0, 0x1715a4ff, 0x248bfec8, 0x329ce87d,
    0x39fdac1d, 0x1eb1a84, 0x2f3daf95, 0x76302d3, 0x2d447b76, 0x19e60794, 0x36912680, 0x263a44a6,
    0x1abb8c2e, 0x1552d982, 0x582d049, 0x1ec86089, 0xa6dabec, 0x36ae8c4d, 0x212929cc, 0x1c58044c,
    0x2abbcd3a, 0x26036dda, 0x2bba5a6f, 0x391e9e23, 0x114990a9, 0xf8cfa13, 0xde43939, 0x9962fac,
    0x2c970e78, 0x19836229, 0x28d530d0, 0x37d65fa5, 0x330d2979, 0x22acafae, 0xd0bd7e7, 0x1c52aad,
    0x23df0e1f, 0x2e6b9097, 0x36606549, 0x2da1612b, 0x4e374e5, 0x31ef1caf, 0x2728dd4f, 0x32f9d3ed,
    0x26f41dc5, 0x1d2fce8, 0x913eeff, 0x3461e202, 0x21407d28, 0x29d13b3, 0x2e03775, 0x233cfd2c,
    0x3b880000, 0x366e3830, 0x371c67d0, 0x2c32c3d, 0x3b1babb3, 0x1a4dae94, 0xddd22f2, 0x1ee46515,
    0x10947bb0, 0x1a7e94e7, 0x2530b82, 0x1f4ac96d, 0x21aa1f30, 0x24725b01, 0x16fc0137, 0x8eb1783,
    0x18a53e3, 0x399ce57c, 0xc4a506b, 0x3424fd2d, 0xe43848a, 0x21a1f86c, 0x4f6d97f, 0x154dbb5a,
    0x20cc73d2, 0x2635267e, 0x36052fb6, 0x1cbf9f77, 0x311a5414, 0x4d973b3, 0x1a5ed634, 0x1f2ffbb3,
    0x10cc32c6, 0x15849226, 0xfcda591, 0x26961dc, 0x2a3e6f57, 0x2bfb05ed, 0x2da3c6c7, 0x31f1d054,
    0xef0f188, 0x22049dd7, 0x12b2cf30, 0x3b1a05b, 0x87ad687, 0x18db5052, 0x2e7c2819, 0x39c2d553,
    0x17a8f1e1, 0xd1c6f69, 0x5279ab7, 0xde69ed5, 0x36a48b1b, 0x998e351, 0x145f22b1, 0x88e2c13,
    0x1493e23b, 0x39b50318, 0x32741101, 0x7261dfe, 0x1a4782d8, 0x38eaec4d, 0x38a7c88b, 0x184b02d4
};

// Inverse Twiddles: psi^{-i} for i=0..127
__constant__ uint32_t g_device_inv_twiddles_bb[DOMAIN_SIZE] = {
    0x1, 0x240085a0, 0x9c24e23a, 0x3990f235, 0x97e7df39, 0x51e33f21, 0xbd6f0dcc, 0x164dce89,
    0xdf651439, 0x52e9bb51, 0xe26b80d7, 0xa81c67b8, 0xf9dd14a6, 0x8d8d9c0a, 0x3b411a10, 0xda1ea31c,
    0x5f1820c8, 0x4e77e3d5, 0x89595a89, 0x86aca177, 0xa5a186b7, 0x1a2b3937, 0x5c14f09f, 0x913cea47,
    0xe460b959, 0xa34a2c0, 0x17a4dc2c, 0x20851df1, 0x79aeb64e, 0xc5913f17, 0x98472985, 0x21015b11,
    0x9c24e23a, 0x697263f7, 0x5f1820c8, 0xa9d94f, 0x97e7df39, 0xe56b6cce, 0xbd6f0dcc, 0xe9b2316a,
    0xdf651439, 0x7d5149b3, 0xe26b80d7, 0x2ea44e54, 0xf9dd14a6, 0x316ec0ea, 0x3b411a10, 0x71d753db,
    0x5f1820c8, 0x7000000, 0x89595a89, 0x71edbfb9, 0xa5a186b7, 0x5cc1a934, 0x5c14f09f, 0x6e53159a,
    0xe460b959, 0x84c3c5e2, 0x17a4dc2c, 0x3c1e6b24, 0x79aeb64e, 0x25a5398d, 0x98472985, 0xd1ae4a36,
    0x9c24e23a, 0x998ff863, 0x5f1820c8, 0x3aedb8f6, 0x97e7df39, 0x41fcb27a, 0xbd6f0dcc, 0x2e38ac5f,
    0xdf651439, 0x5d70079e, 0xe26b80d7, 0x67848599, 0xf9dd14a6, 0xd15ac674, 0x3b411a10, 0xf7b78d29,
    0x5f1820c8, 0xf7000000, 0x89595a89, 0x86255e77, 0xa5a186b7, 0xfc9305c0, 0x5c14f09f, 0x451e1b69,
    0xe460b959, 0x13c79a78, 0x17a4dc2c, 0x4377eb82, 0x79aeb64e, 0x3e92aff6, 0x98472985, 0x7bb49883,
    0x9c24e23a, 0xc5913f17, 0x5f1820c8, 0xd2ff7a61, 0x97e7df39, 0xb5034d87, 0xbd6f0dcc, 0x5d70079e,
    0xdf651439, 0x11949333, 0xe26b80d7, 0xa23c8508, 0xf9dd14a6, 0xb86d500b, 0x3b411a10, 0x33b3184f,
    0x5f1820c8, 0x5f1820c8, 0x89595a89, 0x797b601e, 0xa5a186b7, 0x3cde74dc, 0x5c14f09f, 0x6cff9bac,
    0xe460b959, 0x4590fffa, 0x17a4dc2c, 0xcfca8121, 0x79aeb64e, 0x42e9aa9a, 0x98472985, 0xfc5b8523
};

__constant__ uint32_t g_device_inv_twiddles_kb[DOMAIN_SIZE] = {
    0x0, 0x366e3830, 0x2abbcd3a, 0x399ce57c, 0x18a53e3, 0x2f85b0f, 0x1eb1a84, 0x2e6b9097,
    0x26f41dc5, 0x17b62c56, 0x2c970e78, 0x3629d9a7, 0x248bfec8, 0x1ca39aeb, 0xa6dabec, 0x233cfd2c,
    0x39fdac1d, 0x37b15c62, 0x76302d3, 0x2bd9e0c8, 0x2d447b76, 0x26036dda, 0x19dde0d0, 0x2728dd4f,
    0x4e374e5, 0x8de3e71, 0x28d530d0, 0x36cc1326, 0x2daadd0e, 0x21096b19, 0x114990a9, 0x3461e202,
    0x2abbcd3a, 0x1ee46515, 0x39fdac1d, 0x2e6e39c7, 0x18a53e3, 0x3b1babb3, 0x1eb1a84, 0x2e946e96,
    0x26f41dc5, 0xddd22f2, 0x2c970e78, 0x28ba7e50, 0x248bfec8, 0x1a7e94e7, 0xa6dabec, 0x2e03775,
    0x39fdac1d, 0x3b880000, 0x76302d3, 0x27921c4f, 0x2d447b76, 0x2bba5a6f, 0x19dde0d0, 0x2eda8ea6,
    0x4e374e5, 0x1b7c0dbc, 0x28d530d0, 0x347ddcd8, 0x2daadd0e, 0x3934f47e, 0x114990a9, 0x37ddfde4,
    0x2abbcd3a, 0x10947bb0, 0x39fdac1d, 0x2a0c7ced, 0x18a53e3, 0x46b9830, 0x1eb1a84, 0x2a33c495,
    0x26f41dc5, 0x2af38450, 0x2c970e78, 0x25ff7d18, 0x248bfec8, 0x2530b82, 0xa6dabec, 0x582d049,
    0x39fdac1d, 0x3b880000, 0x76302d3, 0x1a841f38, 0x2d447b76, 0x391e9e23, 0x19dde0d0, 0x276f62cd,
    0x4e374e5, 0x2c24b94c, 0x28d530d0, 0x2bea1c61, 0x2daadd0e, 0x1c3d3693, 0x114990a9, 0x21a467d,
    0x2abbcd3a, 0x21096b19, 0x39fdac1d, 0x519c7d0, 0x18a53e3, 0x6c544d, 0x1eb1a84, 0x2af38450,
    0x26f41dc5, 0x213a516c, 0x2c970e78, 0xc4a506b, 0x248bfec8, 0x1f4ac96d, 0xa6dabec, 0x36ae8c4d,
    0x39fdac1d, 0x39fdac1d, 0x76302d3, 0x1552d982, 0x2d447b76, 0x38c4d3c3, 0x19dde0d0, 0x1715a4ff,
    0x4e374e5, 0x1abb8c2e, 0x28d530d0, 0xf8cfa13, 0x2daadd0e, 0x329ce87d, 0x114990a9, 0x1c58044c
};

// Coset Powers: psi^i for i=0..63
__constant__ uint32_t g_device_coset_powers_bb[POLY_DEGREE] = {
    0x1, 0xd2ff7a61, 0x80335dfe, 0x3cde74dc, 0xb5034d87, 0x11949333, 0x79aeb64e, 0x8d8d9c0a,
    0x5d70079e, 0xc5913f17, 0x25a5398d, 0x3e92aff6, 0x5c14f09f, 0x6cff9bac, 0xf9dd14a6, 0x42e9aa9a,
    0x5f1820c8, 0xbd6f0dcc, 0x54c37af9, 0x89595a89, 0xa5a186b7, 0x6d490d2d, 0xf7133592, 0xed807a9c,
    0x4590fffa, 0x797b601e, 0xf7b78d29, 0xb93b5f7d, 0x3b411a10, 0x33b3184f, 0x1820d679, 0xfc5b8523,
    0x9c24e23a, 0x1a2b3937, 0x5cc1a934, 0xfc9305c0, 0x98472985, 0xcfca8121, 0xe8f26079, 0xb9049dcc,
    0xe26b80d7, 0x57c34fd7, 0x17a4dc2c, 0xcee70043, 0x677ba342, 0x97393de2, 0x6cbe4058, 0x8b7c2dd6,
    0xad9cf002, 0x164dce89, 0xa1d94839, 0x232fa802, 0xe460b959, 0xad73ea1e, 0x913cea47, 0xcdaa57a8,
    0xdf651439, 0x19415948, 0x422ec536, 0x21015b11, 0xf43cbeca, 0x3c491d4c, 0x71d753db, 0xda1ea31c
};

__constant__ uint32_t g_device_coset_powers_kb[POLY_DEGREE] = {
    0x0, 0x519c7d0, 0x46b9830, 0x38c4d3c3, 0x6c544d, 0x213a516c, 0x2daadd0e, 0x1ca39aeb,
    0x2af38450, 0x21096b19, 0x3934f47e, 0x1c3d3693, 0x19dde0d0, 0x1715a4ff, 0x248bfec8, 0x329ce87d,
    0x39fdac1d, 0x1eb1a84, 0x2f3daf95, 0x76302d3, 0x2d447b76, 0x19e60794, 0x36912680, 0x263a44a6,
    0x1abb8c2e, 0x1552d982, 0x582d049, 0x1ec86089, 0xa6dabec, 0x36ae8c4d, 0x212929cc, 0x1c58044c,
    0x2abbcd3a, 0x26036dda, 0x2bba5a6f, 0x391e9e23, 0x114990a9, 0xf8cfa13, 0xde43939, 0x9962fac,
    0x2c970e78, 0x19836229, 0x28d530d0, 0x37d65fa5, 0x330d2979, 0x22acafae, 0xd0bd7e7, 0x1c52aad,
    0x23df0e1f, 0x2e6b9097, 0x36606549, 0x2da1612b, 0x4e374e5, 0x31ef1caf, 0x2728dd4f, 0x32f9d3ed,
    0x26f41dc5, 0x1d2fce8, 0x913eeff, 0x3461e202, 0x21407d28, 0x29d13b3, 0x2e03775, 0x233cfd2c
};

// Inverse Coset Powers: psi^{-i} for i=0..63
__constant__ uint32_t g_device_inv_coset_powers_bb[POLY_DEGREE] = {
    0x00000001, 0x1ce15ce5, 0x8528ac26, 0xbab6e2b5, 0x02c34137, 0xd5fea4f0, 0xb4d13acb, 0xddbea6b9,
    0x179aebc8, 0x2955a859, 0x65c315ba, 0x498c15e3, 0x129f46a8, 0xd3d057ff, 0x5526b7c8, 0xe0b23178,
    0x49630fff, 0x6b83d22b, 0x8a41bfa9, 0x5fc6c21f, 0x8f845cbf, 0x2818ffbe, 0xdf5b23d5, 0x9f3cb02a,
    0x14947f2a, 0x3dfb6235, 0x0e0d9f88, 0x27357ee0, 0x5eb8d67c, 0xfa6cfa41, 0x9a3e56cd, 0xdcd4c6ca,
    0x5adb1dc7, 0xfaa47ade, 0xdedf2988, 0xc34ce7b2, 0xbbbee5f1, 0x3dc4a084, 0xff4872d8, 0x7d849fe3,
    0xb16f0007, 0x097f8565, 0xffecca6f, 0x89b6f2d4, 0x515e794a, 0x6da6a578, 0xa23c8508, 0x3990f235,
    0x97e7df39, 0xb4165567, 0xfd22eb5b, 0x8a006455, 0x9aeb0f62, 0xb86d500b, 0xd15ac674, 0x316ec0ea,
    0x998ff863, 0x697263f7, 0x7d5149b3, 0xe56b6cce, 0x41fcb27a, 0xba218b25, 0x76cca203, 0x240085a0
};

__constant__ uint32_t g_device_inv_coset_powers_kb[POLY_DEGREE] = {
    0x00000000, 0x184b02d4, 0x38a7c88b, 0x38eaec4d, 0x1a4782d8, 0x07261dfe, 0x32741101, 0x39b50318,
    0x1493e23b, 0x088e2c13, 0x145f22b1, 0x0998e351, 0x36a48b1b, 0x0de69ed5, 0x05279ab7, 0x0d1c6f69,
    0x17a8f1e1, 0x39c2d553, 0x2e7c2819, 0x18db5052, 0x087ad687, 0x03b1a05b, 0x12b2cf30, 0x22049dd7,
    0x0ef0f188, 0x31f1d054, 0x2da3c6c7, 0x2bfb05ed, 0x2a3e6f57, 0x026961dc, 0x0fcda591, 0x15849226,
    0x10cc32c6, 0x1f2ffbb3, 0x1a5ed634, 0x04d973b3, 0x311a5414, 0x1cbf9f77, 0x36052fb6, 0x2635267e,
    0x20cc73d2, 0x154dbb5a, 0x04f6d97f, 0x21a1f86c, 0x0e43848a, 0x3424fd2d, 0x0c4a506b, 0x399ce57c,
    0x018a53e3, 0x08eb1783, 0x16fc0137, 0x24725b01, 0x21aa1f30, 0x1f4ac96d, 0x02530b82, 0x1a7e94e7,
    0x10947bb0, 0x1ee46515, 0x0ddd22f2, 0x1a4dae94, 0x3b1babb3, 0x02c32c3d, 0x371c67d0, 0x366e3830
};

__constant__ uint32_t g_device_inv_N_lo;
__constant__ uint32_t g_device_inv_N_hi;

// ============ Bit-reversal ============

__device__ __forceinline__ int bit_reverse_6(int idx) {
    int result = 0;
    result |= (idx & 1) << 5;
    result |= (idx & 2) << 3;
    result |= (idx & 4) << 1;
    result |= (idx & 8) >> 1;
    result |= (idx & 16) >> 3;
    result |= (idx & 32) >> 5;
    return result;
}

// ============ Forward NTT (DIT) Kernel ============

__global__ void forward_ntt_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    int batch_size)
{
    int poly_idx = blockIdx.x;
    if (poly_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int base = poly_idx * POLY_DEGREE * LIMBS_COUNT;
    
    __shared__ uint32_t shmem[POLY_DEGREE * LIMBS_COUNT];
    
    // Step 1: Load (natural order)
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        shmem[i * LIMBS_COUNT] = input[base + i * LIMBS_COUNT];
        shmem[i * LIMBS_COUNT + 1] = input[base + i * LIMBS_COUNT + 1];
    }
    __syncthreads();
    
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        int rev = bit_reverse_6(i);
        if (rev > i) {
            uint64_t vi = load_value(shmem, i);
            uint64_t vr = load_value(shmem, rev);
            store_value(shmem, i, vr);
            store_value(shmem, rev, vi);
        }
    }
    __syncthreads();
    
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        int rev = bit_reverse_6(i);
        if (rev > i) {
            uint64_t vi = load_value(shmem, i);
            uint64_t vr = load_value(shmem, rev);
            store_value(shmem, i, vr);
            store_value(shmem, rev, vi);
        }
    }
    __syncthreads();
        
    // Step 2: Coset multiplication (natural order)
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        uint64_t val = load_value(shmem, i);
        uint64_t coset = load_constant_pair(g_device_coset_powers_bb, g_device_coset_powers_kb, i);
        val = mod_mul64(val, coset);
        store_value(shmem, i, val);
    }
    __syncthreads();
    
    // Step 3: Bit-reverse to match DIT input ordering
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        int rev = bit_reverse_6(i);
        if (rev > i) {
            uint64_t vi = load_value(shmem, i);
            uint64_t vr = load_value(shmem, rev);
            store_value(shmem, i, vr);
            store_value(shmem, rev, vi);
        }
    }
    __syncthreads();
    
    // Step 4: Radix-2 DIT NTT (R→N)
    for (int len = 2; len <= POLY_DEGREE; len <<= 1) {
        int half = len >> 1;
        int step = DOMAIN_SIZE / len;
        
        for (int k = tid; k < POLY_DEGREE / 2; k += 32) {
            int group = k / half;
            int j = k % half;
                int idx1 = group * len + j;
                int idx2 = idx1 + half;
            if (idx2 >= POLY_DEGREE) continue;
            
            uint64_t u = load_value(shmem, idx1);
            uint64_t v = load_value(shmem, idx2);
                
                if (j != 0) {
                    int tw_idx = j * step;
                uint64_t tw = load_constant_pair(g_device_twiddles_bb, g_device_twiddles_kb, tw_idx);
                v = mod_mul64(v, tw);
            }
            
            uint64_t sum = mod_add64(u, v);
            uint64_t diff = mod_sub64(u, v);
            store_value(shmem, idx1, sum);
            store_value(shmem, idx2, diff);
                }
        __syncthreads();
    }
    
    // Step 5: Write (bit-reversed order)
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        int rev_i = bit_reverse_6(i);
        output[base + rev_i * LIMBS_COUNT] = shmem[i * LIMBS_COUNT];
        output[base + rev_i * LIMBS_COUNT + 1] = shmem[i * LIMBS_COUNT + 1];
    }
}

// ============ Inverse NTT (DIT) Kernel ============

__global__ void inverse_ntt_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    int batch_size)
{
    int poly_idx = blockIdx.x;
    if (poly_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int base = poly_idx * POLY_DEGREE * LIMBS_COUNT;
    
    __shared__ uint32_t shmem[POLY_DEGREE * LIMBS_COUNT];
    
    // Step 1: Load (natural order input)
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        shmem[i * LIMBS_COUNT] = input[base + i * LIMBS_COUNT];
        shmem[i * LIMBS_COUNT + 1] = input[base + i * LIMBS_COUNT + 1];
    }
    __syncthreads();
    
    // Step 2: Radix-2 DIT INTT (R→N)
    for (int len = 2; len <= POLY_DEGREE; len <<= 1) {
        int half = len >> 1;
        int step = DOMAIN_SIZE / len;
        
        if (tid == 0) {
            for (int block = 0; block < POLY_DEGREE; block += len) {
                for (int j = 0; j < half; ++j) {
                    int idx1 = block + j;
                int idx2 = idx1 + half;
                    uint64_t u = load_value(shmem, idx1);
                    uint64_t v = load_value(shmem, idx2);
                if (j != 0) {
                        int tw_idx = (DOMAIN_SIZE - j * step) & (DOMAIN_SIZE - 1);
                        uint64_t tw = load_constant_pair(g_device_twiddles_bb, g_device_twiddles_kb, tw_idx);
                        v = mod_mul64(v, tw);
                    }
                    uint64_t sum = mod_add64(u, v);
                    uint64_t diff = mod_sub64(u, v);
                    store_value(shmem, idx1, sum);
                    store_value(shmem, idx2, diff);
                }
            }
        }
        __syncthreads();
    }
    
    // Step 3: Divide by N
    uint64_t inv_n = combine_u64(g_device_inv_N_lo, g_device_inv_N_hi);
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        uint64_t val = load_value(shmem, i);
        val = mod_mul64(val, inv_n);
        store_value(shmem, i, val);
    }
    __syncthreads();
    
    // Step 4: Inverse coset multiplication (natural order)
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        uint64_t val = load_value(shmem, i);
        uint64_t inv_coset = load_constant_pair(g_device_inv_coset_powers_bb, g_device_inv_coset_powers_kb, i);
        val = mod_mul64(val, inv_coset);
        store_value(shmem, i, val);
    }
    __syncthreads();
    
    // Step 5: Write (natural order output)
    for (int i = tid; i < POLY_DEGREE; i += 32) {
        output[base + i * LIMBS_COUNT] = shmem[i * LIMBS_COUNT];
        output[base + i * LIMBS_COUNT + 1] = shmem[i * LIMBS_COUNT + 1];
    }
}

// ============ 主机端函数 ============

static uint64_t mod_inverse_u64(uint64_t a, uint64_t mod) {
    int64_t t = 0;
    int64_t new_t = 1;
    int64_t r = static_cast<int64_t>(mod);
    int64_t new_r = static_cast<int64_t>(a % mod);
    
    while (new_r != 0) {
        int64_t quotient = r / new_r;
        int64_t tmp = new_t;
        new_t = t - quotient * new_t;
        t = tmp;

        tmp = new_r;
        new_r = r - quotient * new_r;
        r = tmp;
}

    if (r > 1) {
        throw std::runtime_error("mod_inverse_u64: inverse does not exist");
    }
    if (t < 0) {
        t += static_cast<int64_t>(mod);
    }
    return static_cast<uint64_t>(t);
}

extern "C" {

void custom_ntt_init_cuda_with_roots(const uint32_t*, const uint32_t*)
{
    uint64_t inv_n = mod_inverse_u64(POLY_DEGREE, MODULUS_Q);
    uint32_t inv_lo, inv_hi;
    split_u64(inv_n, inv_lo, inv_hi);
    cudaMemcpyToSymbol(g_device_inv_N_lo, &inv_lo, sizeof(uint32_t));
    cudaMemcpyToSymbol(g_device_inv_N_hi, &inv_hi, sizeof(uint32_t));
}

int custom_ntt_forward_cuda(const void* input, void* output, int batch_size)
{
    size_t data_size = batch_size * POLY_DEGREE * LIMBS_COUNT * sizeof(uint32_t);
    
    uint32_t *d_input, *d_output;
    cudaError_t err = cudaMalloc(&d_input, data_size);
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] cudaMalloc d_input failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    err = cudaMalloc(&d_output, data_size);
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] cudaMalloc d_output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return -1;
    }
    
    err = cudaMemcpy(d_input, input, data_size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] cudaMemcpy input failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }
    
    dim3 grid(batch_size);
    dim3 block(32);
    forward_ntt_kernel<<<grid, block>>>(d_input, d_output, batch_size);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] Kernel sync error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }
    
    err = cudaMemcpy(output, d_output, data_size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] cudaMemcpy output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

int custom_ntt_inverse_cuda(const void* input, void* output, int batch_size)
{
    size_t data_size = batch_size * POLY_DEGREE * LIMBS_COUNT * sizeof(uint32_t);
    
    uint32_t *d_input, *d_output;
    cudaError_t err = cudaMalloc(&d_input, data_size);
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] INTT: cudaMalloc d_input failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    err = cudaMalloc(&d_output, data_size);
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] INTT: cudaMalloc d_output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return -1;
    }
    
    err = cudaMemcpy(d_input, input, data_size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] INTT: cudaMemcpy input failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }
    
    dim3 grid(batch_size);
    dim3 block(32);
    inverse_ntt_kernel<<<grid, block>>>(d_input, d_output, batch_size);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] INTT: Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] INTT: Kernel sync error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }
    
    err = cudaMemcpy(output, d_output, data_size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "[CUSTOM NTT] INTT: cudaMemcpy output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

void custom_ntt_cleanup_cuda()
{
}

cudaError_t custom_ntt_get_constants(CustomNttConstants* host_consts)
{
    if (host_consts == nullptr) return cudaErrorInvalidValue;
    
    cudaError_t err = cudaMemcpyFromSymbol(
        host_consts->inv_twiddles_bb, g_device_inv_twiddles_bb, sizeof(host_consts->inv_twiddles_bb));
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyFromSymbol(
        host_consts->inv_twiddles_kb, g_device_inv_twiddles_kb, sizeof(host_consts->inv_twiddles_kb));
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyFromSymbol(host_consts->coset_bb, g_device_coset_powers_bb, sizeof(host_consts->coset_bb));
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyFromSymbol(host_consts->coset_kb, g_device_coset_powers_kb, sizeof(host_consts->coset_kb));
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyFromSymbol(
        host_consts->inv_coset_bb, g_device_inv_coset_powers_bb, sizeof(host_consts->inv_coset_bb));
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyFromSymbol(
        host_consts->inv_coset_kb, g_device_inv_coset_powers_kb, sizeof(host_consts->inv_coset_kb));
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyFromSymbol(&host_consts->inv_N_lo, g_device_inv_N_lo, sizeof(host_consts->inv_N_lo));
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyFromSymbol(&host_consts->inv_N_hi, g_device_inv_N_hi, sizeof(host_consts->inv_N_hi));
    if (err != cudaSuccess) return err;
    
    return cudaSuccess;
}

} // extern "C"
