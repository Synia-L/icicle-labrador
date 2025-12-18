/*****************************************************************************
 * Custom CUDA Matrix Multiplication Implementation
 * 
 * 实现标量域和多项式环的矩阵乘法，参考 CPU 版本逻辑
 * 支持转置标志（a_transposed, b_transposed）
 *****************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include "custom_ntt_constants.h"

// ============ 常量定义 ============
const int POLY_DEGREE = kCustomPolyDegree;  // 64
const int LIMBS_COUNT = kCustomLimbs;       // 2
const uint64_t MODULUS_Q = kCustomModulusQ;

// ============ 设备端辅助函数 ============

__host__ __device__ __forceinline__ uint64_t combine_u64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

__host__ __device__ __forceinline__ void split_u64(uint64_t value, uint32_t& lo, uint32_t& hi) {
    lo = static_cast<uint32_t>(value & 0xffffffffULL);
    hi = static_cast<uint32_t>(value >> 32);
}

__device__ __forceinline__ uint64_t load_field_element(const uint32_t* data, int idx) {
    return combine_u64(data[idx * LIMBS_COUNT], data[idx * LIMBS_COUNT + 1]);
}

__device__ __forceinline__ void store_field_element(uint32_t* data, int idx, uint64_t value) {
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

__device__ __forceinline__ uint64_t mod_mul64(uint64_t a, uint64_t b) {
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return static_cast<uint64_t>(prod % MODULUS_Q);
}

// ============ CUDA Kernels ============

/**
 * @brief 标量矩阵乘法内核（degree = 1）
 * 
 * 计算 C[row, col] = sum_k A[row, k] * B[k, col]
 * 每个线程计算输出矩阵的一个元素
 * 
 * @param mat_a 输入矩阵 A (effective_rows_a × effective_cols_a)
 * @param mat_b 输入矩阵 B (effective_rows_b × effective_cols_b)
 * @param mat_out 输出矩阵 C (effective_rows_a × effective_cols_b)
 * @param effective_rows_a A 的有效行数（考虑转置）
 * @param effective_cols_a A 的有效列数（考虑转置）
 * @param effective_cols_b B 的有效列数（考虑转置）
 * @param nof_cols_a A 的原始列数
 * @param nof_cols_b B 的原始列数
 * @param a_transposed A 是否转置
 * @param b_transposed B 是否转置
 */
__global__ void matmul_scalar_kernel(
    const uint32_t* mat_a,
    const uint32_t* mat_b,
    uint32_t* mat_out,
    uint32_t effective_rows_a,
    uint32_t effective_cols_a,
    uint32_t effective_cols_b,
    uint32_t nof_cols_a,
    uint32_t nof_cols_b,
    bool a_transposed,
    bool b_transposed
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= effective_rows_a || col >= effective_cols_b) return;

    uint64_t acc = 0;

    // 内积累加：sum_k A[row, k] * B[k, col]
    for (uint32_t k = 0; k < effective_cols_a; ++k) {
        // 获取 A[row, k] 的索引
        uint32_t a_idx;
        if (a_transposed) {
            // A^T[row, k] = A[k, row]
            a_idx = k * nof_cols_a + row;
        } else {
            a_idx = row * nof_cols_a + k;
        }

        // 获取 B[k, col] 的索引
        uint32_t b_idx;
        if (b_transposed) {
            // B^T[k, col] = B[col, k]
            b_idx = col * nof_cols_b + k;
        } else {
            b_idx = k * nof_cols_b + col;
        }

        uint64_t a_val = load_field_element(mat_a, a_idx);
        uint64_t b_val = load_field_element(mat_b, b_idx);
        acc = mod_add64(acc, mod_mul64(a_val, b_val));
    }

    uint32_t out_idx = row * effective_cols_b + col;
    store_field_element(mat_out, out_idx, acc);
}

/**
 * @brief 多项式环矩阵乘法内核（degree = POLY_DEGREE）
 * 
 * 每个矩阵元素是一个多项式（POLY_DEGREE 个系数）
 * 计算 C[row, col][d] = sum_k A[row, k][d] * B[k, col][d]  (逐系数相乘)
 * 
 * @param degree 多项式度数（每个元素的系数个数）
 */
__global__ void matmul_poly_kernel(
    const uint32_t* mat_a,
    const uint32_t* mat_b,
    uint32_t* mat_out,
    uint32_t effective_rows_a,
    uint32_t effective_cols_a,
    uint32_t effective_cols_b,
    uint32_t nof_cols_a,
    uint32_t nof_cols_b,
    bool a_transposed,
    bool b_transposed,
    uint32_t degree
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= effective_rows_a || col >= effective_cols_b) return;

    // 每个线程计算 C[row, col] 的所有 degree 个系数
    for (uint32_t d = 0; d < degree; ++d) {
        uint64_t acc = 0;

        // 内积累加
        for (uint32_t k = 0; k < effective_cols_a; ++k) {
            // 获取 A[row, k][d]
            uint32_t a_base_idx;
            if (a_transposed) {
                a_base_idx = k * nof_cols_a + row;
            } else {
                a_base_idx = row * nof_cols_a + k;
            }
            uint32_t a_idx = a_base_idx * degree + d;

            // 获取 B[k, col][d]
            uint32_t b_base_idx;
            if (b_transposed) {
                b_base_idx = col * nof_cols_b + k;
            } else {
                b_base_idx = k * nof_cols_b + col;
            }
            uint32_t b_idx = b_base_idx * degree + d;

            uint64_t a_val = load_field_element(mat_a, a_idx);
            uint64_t b_val = load_field_element(mat_b, b_idx);
            acc = mod_add64(acc, mod_mul64(a_val, b_val));
        }

        uint32_t out_idx = (row * effective_cols_b + col) * degree + d;
        store_field_element(mat_out, out_idx, acc);
    }
}

// ============ 主机端接口 ============

/**
 * @brief 标量矩阵乘法主机接口
 * 
 * @param mat_a_host 输入矩阵 A (行主序, uint32_t[nof_rows_a * nof_cols_a * LIMBS_COUNT])
 * @param nof_rows_a A 的行数
 * @param nof_cols_a A 的列数
 * @param mat_b_host 输入矩阵 B (行主序, uint32_t[nof_rows_b * nof_cols_b * LIMBS_COUNT])
 * @param nof_rows_b B 的行数
 * @param nof_cols_b B 的列数
 * @param mat_out_host 输出矩阵 C (行主序)
 * @param a_transposed A 是否转置
 * @param b_transposed B 是否转置
 * @return 0 成功，-1 失败
 */
extern "C" int custom_matmul_scalar_cuda(
    const void* mat_a_host,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const void* mat_b_host,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    void* mat_out_host,
    bool a_transposed,
    bool b_transposed
) {
    try {
        // 计算有效维度
        uint32_t effective_rows_a = a_transposed ? nof_cols_a : nof_rows_a;
        uint32_t effective_cols_a = a_transposed ? nof_rows_a : nof_cols_a;
        uint32_t effective_rows_b = b_transposed ? nof_cols_b : nof_rows_b;
        uint32_t effective_cols_b = b_transposed ? nof_rows_b : nof_cols_b;

        if (effective_cols_a != effective_rows_b) {
            std::cerr << "[CUSTOM MATMUL] Error: inner dimensions do not match" << std::endl;
            return -1;
        }

        // 计算内存大小
        size_t size_a = nof_rows_a * nof_cols_a * LIMBS_COUNT * sizeof(uint32_t);
        size_t size_b = nof_rows_b * nof_cols_b * LIMBS_COUNT * sizeof(uint32_t);
        size_t size_out = effective_rows_a * effective_cols_b * LIMBS_COUNT * sizeof(uint32_t);

        // 分配设备内存
        uint32_t *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, size_a);
        cudaMalloc(&d_b, size_b);
        cudaMalloc(&d_out, size_out);

        // 拷贝输入到设备
        cudaMemcpy(d_a, mat_a_host, size_a, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, mat_b_host, size_b, cudaMemcpyHostToDevice);

        // 配置 kernel
        dim3 block(16, 16);
        dim3 grid(
            (effective_cols_b + block.x - 1) / block.x,
            (effective_rows_a + block.y - 1) / block.y
        );

        // 启动 kernel
        matmul_scalar_kernel<<<grid, block>>>(
            d_a, d_b, d_out,
            effective_rows_a, effective_cols_a, effective_cols_b,
            nof_cols_a, nof_cols_b,
            a_transposed, b_transposed
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM MATMUL] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_out);
            return -1;
        }

        // 同步并拷贝结果
        cudaDeviceSynchronize();
        cudaMemcpy((void*)mat_out_host, d_out, size_out, cudaMemcpyDeviceToHost);

        // 释放内存
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM MATMUL] Exception: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief 多项式环矩阵乘法主机接口
 * 
 * @param degree 多项式度数（每个矩阵元素的系数个数）
 */
extern "C" int custom_matmul_poly_cuda(
    const void* mat_a_host,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const void* mat_b_host,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    void* mat_out_host,
    bool a_transposed,
    bool b_transposed,
    uint32_t degree
) {
    try {
        // 计算有效维度
        uint32_t effective_rows_a = a_transposed ? nof_cols_a : nof_rows_a;
        uint32_t effective_cols_a = a_transposed ? nof_rows_a : nof_cols_a;
        uint32_t effective_rows_b = b_transposed ? nof_cols_b : nof_rows_b;
        uint32_t effective_cols_b = b_transposed ? nof_rows_b : nof_cols_b;

        if (effective_cols_a != effective_rows_b) {
            std::cerr << "[CUSTOM MATMUL POLY] Error: inner dimensions do not match" << std::endl;
            return -1;
        }

        // 计算内存大小（每个元素有 degree 个系数，每个系数 LIMBS_COUNT 个 uint32）
        size_t size_a = nof_rows_a * nof_cols_a * degree * LIMBS_COUNT * sizeof(uint32_t);
        size_t size_b = nof_rows_b * nof_cols_b * degree * LIMBS_COUNT * sizeof(uint32_t);
        size_t size_out = effective_rows_a * effective_cols_b * degree * LIMBS_COUNT * sizeof(uint32_t);

        // 分配设备内存
        uint32_t *d_a, *d_b, *d_out;
        cudaMalloc(&d_a, size_a);
        cudaMalloc(&d_b, size_b);
        cudaMalloc(&d_out, size_out);

        // 拷贝输入到设备
        cudaMemcpy(d_a, mat_a_host, size_a, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, mat_b_host, size_b, cudaMemcpyHostToDevice);

        // 配置 kernel（每个线程处理一个矩阵元素的所有系数）
        dim3 block(16, 16);
        dim3 grid(
            (effective_cols_b + block.x - 1) / block.x,
            (effective_rows_a + block.y - 1) / block.y
        );

        // 启动 kernel
        matmul_poly_kernel<<<grid, block>>>(
            d_a, d_b, d_out,
            effective_rows_a, effective_cols_a, effective_cols_b,
            nof_cols_a, nof_cols_b,
            a_transposed, b_transposed,
            degree
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM MATMUL POLY] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_out);
            return -1;
        }

        // 同步并拷贝结果
        cudaDeviceSynchronize();
        cudaMemcpy((void*)mat_out_host, d_out, size_out, cudaMemcpyDeviceToHost);

        // 释放内存
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM MATMUL POLY] Exception: " << e.what() << std::endl;
        return -1;
    }
}
