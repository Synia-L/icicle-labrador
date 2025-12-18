/*****************************************************************************
 * Custom CUDA Vector Operations Implementation
 * 
 * 实现四个向量操作：scalar_mul_vec、vector_add、vector_mul、vector_sum
 * 参考 CPU 版本逻辑（icicle/backend/cpu/src/field/cpu_vec_ops.cpp）
 *****************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "custom_ntt_constants.h"

// ============ 常量定义 ============
const int LIMBS_COUNT_VEC = kCustomLimbs;       // 2
const uint64_t MODULUS_Q_VEC = kCustomModulusQ;

// ============ 设备端辅助函数 ============

__host__ __device__ __forceinline__ uint64_t combine_u64_vec(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

__host__ __device__ __forceinline__ void split_u64_vec(uint64_t value, uint32_t& lo, uint32_t& hi) {
    lo = static_cast<uint32_t>(value & 0xffffffffULL);
    hi = static_cast<uint32_t>(value >> 32);
}

__device__ __forceinline__ uint64_t load_element_vec(const uint32_t* data, uint64_t idx) {
    return combine_u64_vec(data[idx * LIMBS_COUNT_VEC], data[idx * LIMBS_COUNT_VEC + 1]);
}

__device__ __forceinline__ void store_element_vec(uint32_t* data, uint64_t idx, uint64_t value) {
    uint32_t lo, hi;
    split_u64_vec(value, lo, hi);
    data[idx * LIMBS_COUNT_VEC] = lo;
    data[idx * LIMBS_COUNT_VEC + 1] = hi;
}

__host__ __device__ __forceinline__ uint64_t mod_add64_vec(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    if (sum >= MODULUS_Q_VEC || sum < a) sum -= MODULUS_Q_VEC;
    return sum;
}

__device__ __forceinline__ uint64_t mod_mul64_vec(uint64_t a, uint64_t b) {
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return static_cast<uint64_t>(prod % MODULUS_Q_VEC);
}

// ============ CUDA Kernels ============

/**
 * @brief 标量乘向量内核 (scalar * vector[i * stride])
 * 
 * @param scalar 标量值
 * @param vec 输入向量
 * @param output 输出向量
 * @param size 向量元素个数
 * @param stride 步长（支持列批处理）
 */
__global__ void scalar_mul_vec_kernel(
    const uint32_t* scalar,
    const uint32_t* vec,
    uint32_t* output,
    uint64_t size,
    uint64_t stride,
    uint32_t degree
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    uint64_t scalar_val = load_element_vec(scalar, 0);

    // 逐系数处理（degree=1 时退化为标量）
    for (uint32_t d = 0; d < degree; ++d) {
        uint64_t vec_val = load_element_vec(vec, idx * stride * degree + d);
        uint64_t result = mod_mul64_vec(scalar_val, vec_val);
        store_element_vec(output, idx * stride * degree + d, result);
    }
}

/**
 * @brief 向量加法内核 (vec_a + vec_b)
 * 
 * @param vec_a 输入向量 A
 * @param vec_b 输入向量 B
 * @param output 输出向量
 * @param size 向量元素个数
 */
__global__ void vector_add_kernel(
    const uint32_t* vec_a,
    const uint32_t* vec_b,
    uint32_t* output,
    uint64_t size,
    uint32_t degree
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    for (uint32_t d = 0; d < degree; ++d) {
        uint64_t a_val = load_element_vec(vec_a, idx * degree + d);
        uint64_t b_val = load_element_vec(vec_b, idx * degree + d);
        uint64_t result = mod_add64_vec(a_val, b_val);
        store_element_vec(output, idx * degree + d, result);
    }
}

/**
 * @brief 向量乘法内核 (vec_a * vec_b，逐元素相乘)
 * 
 * @param vec_a 输入向量 A
 * @param vec_b 输入向量 B
 * @param output 输出向量
 * @param size 向量元素个数
 * @param degree 多项式度数
 */
__global__ void vector_mul_kernel(
    const uint32_t* vec_a,
    const uint32_t* vec_b,
    uint32_t* output,
    uint64_t size,
    uint32_t degree
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    for (uint32_t d = 0; d < degree; ++d) {
        uint64_t a_val = load_element_vec(vec_a, idx * degree + d);
        uint64_t b_val = load_element_vec(vec_b, idx * degree + d);
        uint64_t result = mod_mul64_vec(a_val, b_val);
        store_element_vec(output, idx * degree + d, result);
    }
}

/**
 * @brief 向量求和内核 - 两阶段归约
 * 
 * 第一阶段：每个块内使用共享内存归约
 * 第二阶段：在主机端累加各块的部分和
 * 
 * @param vec 输入向量
 * @param partial_sums 各块的部分和输出（长度 = gridDim.x）
 * @param size 向量元素个数
 * @param stride 步长
 */
__global__ void vector_sum_kernel(
    const uint32_t* vec,
    uint32_t* partial_sums,
    uint64_t size,
    uint64_t stride,
    uint32_t degree
) {
    extern __shared__ uint64_t sdata[];

    uint32_t coeff = blockIdx.y;                 // 处理的系数索引
    uint64_t tid = threadIdx.x;
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; // 元素索引

    // 每个线程加载一个元素的指定系数
    uint64_t sum = 0;
    if (idx < size) {
        uint64_t offset = idx * stride * degree + coeff;
        sum = load_element_vec(vec, offset);
    }
    sdata[tid] = sum;
    __syncthreads();

    // 块内归约
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mod_add64_vec(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 第一个线程写出块的部分和
    if (tid == 0) {
        uint64_t partial_idx = static_cast<uint64_t>(coeff) * gridDim.x + blockIdx.x;
        store_element_vec(partial_sums, partial_idx, sdata[0]);
    }
}

// ============ 主机端接口 ============

/**
 * @brief 标量乘向量主机接口
 * 
 * @param scalar_host 标量（单个元素）
 * @param vec_host 输入向量
 * @param output_host 输出向量
 * @param size 向量元素个数
 * @param stride 步长
 * @return 0 成功，-1 失败
 */
extern "C" int custom_scalar_mul_vec_cuda(
    const void* scalar_host,
    const void* vec_host,
    void* output_host,
    uint64_t size,
    uint64_t stride,
    uint32_t degree
) {
    try {
        size_t scalar_bytes = LIMBS_COUNT_VEC * sizeof(uint32_t);
        size_t vec_bytes = size * stride * degree * LIMBS_COUNT_VEC * sizeof(uint32_t);

        // 分配设备内存
        uint32_t *d_scalar, *d_vec, *d_output;
        cudaMalloc(&d_scalar, scalar_bytes);
        cudaMalloc(&d_vec, vec_bytes);
        cudaMalloc(&d_output, vec_bytes);

        // 拷贝输入到设备
        cudaMemcpy(d_scalar, scalar_host, scalar_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec, vec_host, vec_bytes, cudaMemcpyHostToDevice);

        // 配置 kernel
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;

        // 启动 kernel
        scalar_mul_vec_kernel<<<grid_size, block_size>>>(
            d_scalar, d_vec, d_output, size, stride, degree
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM VEC OPS] scalar_mul_vec kernel failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_scalar); cudaFree(d_vec); cudaFree(d_output);
            return -1;
        }

        // 同步并拷贝结果
        cudaDeviceSynchronize();
        cudaMemcpy((void*)output_host, d_output, vec_bytes, cudaMemcpyDeviceToHost);

        // 释放内存
        cudaFree(d_scalar);
        cudaFree(d_vec);
        cudaFree(d_output);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM VEC OPS] scalar_mul_vec exception: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief 向量加法主机接口
 * 
 * @param vec_a_host 输入向量 A
 * @param vec_b_host 输入向量 B
 * @param output_host 输出向量
 * @param size 向量元素个数
 * @return 0 成功，-1 失败
 */
extern "C" int custom_vector_add_cuda(
    const void* vec_a_host,
    const void* vec_b_host,
    void* output_host,
    uint64_t size,
    uint32_t degree
) {
    try {
        size_t vec_bytes = size * degree * LIMBS_COUNT_VEC * sizeof(uint32_t);

        // 分配设备内存
        uint32_t *d_vec_a, *d_vec_b, *d_output;
        cudaMalloc(&d_vec_a, vec_bytes);
        cudaMalloc(&d_vec_b, vec_bytes);
        cudaMalloc(&d_output, vec_bytes);

        // 拷贝输入到设备
        cudaMemcpy(d_vec_a, vec_a_host, vec_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec_b, vec_b_host, vec_bytes, cudaMemcpyHostToDevice);

        // 配置 kernel
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;

        // 启动 kernel
        vector_add_kernel<<<grid_size, block_size>>>(
            d_vec_a, d_vec_b, d_output, size, degree
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM VEC OPS] vector_add kernel failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_vec_a); cudaFree(d_vec_b); cudaFree(d_output);
            return -1;
        }

        // 同步并拷贝结果
        cudaDeviceSynchronize();
        cudaMemcpy((void*)output_host, d_output, vec_bytes, cudaMemcpyDeviceToHost);

        // 释放内存
        cudaFree(d_vec_a);
        cudaFree(d_vec_b);
        cudaFree(d_output);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM VEC OPS] vector_add exception: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief 向量乘法主机接口 (逐元素相乘)
 * 
 * @param vec_a_host 输入向量 A
 * @param vec_b_host 输入向量 B
 * @param output_host 输出向量
 * @param size 向量元素个数
 * @param degree 多项式度数
 * @return 0 成功，-1 失败
 */
extern "C" int custom_vector_mul_cuda(
    const void* vec_a_host,
    const void* vec_b_host,
    void* output_host,
    uint64_t size,
    uint32_t degree
) {
    try {
        size_t vec_bytes = size * degree * LIMBS_COUNT_VEC * sizeof(uint32_t);

        // 分配设备内存
        uint32_t *d_vec_a, *d_vec_b, *d_output;
        cudaMalloc(&d_vec_a, vec_bytes);
        cudaMalloc(&d_vec_b, vec_bytes);
        cudaMalloc(&d_output, vec_bytes);

        // 拷贝输入到设备
        cudaMemcpy(d_vec_a, vec_a_host, vec_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec_b, vec_b_host, vec_bytes, cudaMemcpyHostToDevice);

        // 配置 kernel
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;

        // 启动 kernel
        vector_mul_kernel<<<grid_size, block_size>>>(
            d_vec_a, d_vec_b, d_output, size, degree
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM VEC OPS] vector_mul kernel failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_vec_a); cudaFree(d_vec_b); cudaFree(d_output);
            return -1;
        }

        // 同步并拷贝结果
        cudaDeviceSynchronize();
        cudaMemcpy((void*)output_host, d_output, vec_bytes, cudaMemcpyDeviceToHost);

        // 释放内存
        cudaFree(d_vec_a);
        cudaFree(d_vec_b);
        cudaFree(d_output);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM VEC OPS] vector_mul exception: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief 向量求和主机接口
 * 
 * @param vec_host 输入向量
 * @param output_host 输出标量（单个元素）
 * @param size 向量元素个数
 * @param stride 步长
 * @return 0 成功，-1 失败
 */
extern "C" int custom_vector_sum_cuda(
    const void* vec_host,
    void* output_host,
    uint64_t size,
    uint64_t stride,
    uint32_t degree
) {
    try {
        size_t vec_bytes = size * stride * degree * LIMBS_COUNT_VEC * sizeof(uint32_t);

        // 分配设备内存
        uint32_t *d_vec;
        cudaMalloc(&d_vec, vec_bytes);
        cudaMemcpy(d_vec, vec_host, vec_bytes, cudaMemcpyHostToDevice);

        // 第一阶段：块级归约
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        dim3 grid(grid_size, degree, 1);
        size_t partial_bytes = static_cast<size_t>(grid_size) * degree * LIMBS_COUNT_VEC * sizeof(uint32_t);
        uint32_t *d_partial;
        cudaMalloc(&d_partial, partial_bytes);

        size_t shared_mem_size = block_size * sizeof(uint64_t);
        vector_sum_kernel<<<grid, block_size, shared_mem_size>>>(
            d_vec, d_partial, size, stride, degree
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM VEC OPS] vector_sum kernel failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_vec); cudaFree(d_partial);
            return -1;
        }
        cudaDeviceSynchronize();

        // 第二阶段：在主机端累加各块的部分和
        std::vector<uint32_t> partial_sums_host(partial_bytes / sizeof(uint32_t));
        cudaMemcpy(partial_sums_host.data(), d_partial, partial_bytes, cudaMemcpyDeviceToHost);

        uint32_t* output_ptr = static_cast<uint32_t*>(output_host);
        for (uint32_t coeff = 0; coeff < degree; ++coeff) {
            uint64_t final_sum = 0;
            for (int bx = 0; bx < grid_size; ++bx) {
                size_t base = (static_cast<size_t>(coeff) * grid_size + bx) * LIMBS_COUNT_VEC;
                uint64_t partial = combine_u64_vec(
                    partial_sums_host[base],
                    partial_sums_host[base + 1]
                );
                final_sum = mod_add64_vec(final_sum, partial);
            }
            // 写回结果
            split_u64_vec(final_sum, output_ptr[coeff * LIMBS_COUNT_VEC], output_ptr[coeff * LIMBS_COUNT_VEC + 1]);
        }

        // 释放内存
        cudaFree(d_vec);
        cudaFree(d_partial);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM VEC OPS] vector_sum exception: " << e.what() << std::endl;
        return -1;
    }
}
