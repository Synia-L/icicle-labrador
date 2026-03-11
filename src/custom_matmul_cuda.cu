/*****************************************************************************
 * Custom CUDA Matrix Multiplication Implementation
 * 
 * 实现标量域和多项式环的矩阵乘法，参考 CPU 版本逻辑
 * 支持转置标志（a_transposed, b_transposed）
 * 
 * 优化特性（自动启用）：
 * - GPU内存池（避免频繁malloc/free）
 * - 异步传输 + CUDA Streams
 * - Shared memory tiling
 *****************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include "custom_ntt_constants.h"
#include "cuda_memory_pool.h"
#include "cuda_async_ops.h"

using namespace cuda_opt;

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

// Tile size for shared memory optimization
#define TILE_SIZE 32

/**
 * @brief 标量矩阵乘法内核（degree = 1）- Shared Memory Tiling 优化版
 * 
 * 使用共享内存分块优化，减少全局内存访问次数
 * 计算 C[row, col] = sum_k A[row, k] * B[k, col]
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
    // Shared memory tiles for A and B
    __shared__ uint64_t tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ uint64_t tile_b[TILE_SIZE][TILE_SIZE];

    uint32_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    uint32_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;

    uint64_t acc = 0;

    // Loop over tiles
    uint32_t num_tiles = (effective_cols_a + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        // Load tile of A into shared memory
        uint32_t a_row = row;
        uint32_t a_col = tile * TILE_SIZE + tx;
        
        if (a_row < effective_rows_a && a_col < effective_cols_a) {
        uint32_t a_idx;
        if (a_transposed) {
                a_idx = a_col * nof_cols_a + a_row;
            } else {
                a_idx = a_row * nof_cols_a + a_col;
            }
            tile_a[ty][tx] = load_field_element(mat_a, a_idx);
        } else {
            tile_a[ty][tx] = 0;
        }

        // Load tile of B into shared memory
        uint32_t b_row = tile * TILE_SIZE + ty;
        uint32_t b_col = col;
        
        if (b_row < effective_cols_a && b_col < effective_cols_b) {
        uint32_t b_idx;
        if (b_transposed) {
                b_idx = b_col * nof_cols_b + b_row;
            } else {
                b_idx = b_row * nof_cols_b + b_col;
            }
            tile_b[ty][tx] = load_field_element(mat_b, b_idx);
        } else {
            tile_b[ty][tx] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (uint32_t k = 0; k < TILE_SIZE; ++k) {
            acc = mod_add64(acc, mod_mul64(tile_a[ty][k], tile_b[k][tx]));
    }

        __syncthreads();
    }

    // Write result
    if (row < effective_rows_a && col < effective_cols_b) {
    uint32_t out_idx = row * effective_cols_b + col;
    store_field_element(mat_out, out_idx, acc);
    }
}

/**
 * @brief 多项式环矩阵乘法内核（degree = POLY_DEGREE）- Shared Memory 优化版
 * 
 * 每个矩阵元素是一个多项式（degree 个系数）
 * 计算 C[row, col][d] = sum_k A[row, k][d] * B[k, col][d]  (逐系数相乘)
 * 使用共享内存缓存输入tiles，减少全局内存访问
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
    // 使用较小的tile size以适应多项式的额外内存需求
    const uint32_t POLY_TILE = 16;
    
    // 每个线程负责输出矩阵的一个元素（包含degree个系数）
    uint32_t row = blockIdx.y * POLY_TILE + threadIdx.y;
    uint32_t col = blockIdx.x * POLY_TILE + threadIdx.x;

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;

    // 累加器数组，存储当前输出元素的所有系数
    uint64_t acc[64];  // 最大支持degree=64
    for (uint32_t d = 0; d < degree; ++d) {
        acc[d] = 0;
    }

    // Loop over tiles in k dimension
    uint32_t num_tiles = (effective_cols_a + POLY_TILE - 1) / POLY_TILE;
    
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        // 对每个系数，分别处理（避免shared memory过大）
        for (uint32_t d = 0; d < degree; ++d) {
            __shared__ uint64_t tile_a[POLY_TILE][POLY_TILE];
            __shared__ uint64_t tile_b[POLY_TILE][POLY_TILE];
            
            // Load tile of A[.][.][d] into shared memory
            uint32_t a_row = row;
            uint32_t a_col = tile * POLY_TILE + tx;
            
            if (a_row < effective_rows_a && a_col < effective_cols_a) {
            uint32_t a_base_idx;
            if (a_transposed) {
                    a_base_idx = a_col * nof_cols_a + a_row;
                } else {
                    a_base_idx = a_row * nof_cols_a + a_col;
                }
                tile_a[ty][tx] = load_field_element(mat_a, a_base_idx * degree + d);
            } else {
                tile_a[ty][tx] = 0;
            }

            // Load tile of B[.][.][d] into shared memory
            uint32_t b_row = tile * POLY_TILE + ty;
            uint32_t b_col = col;

            if (b_row < effective_cols_a && b_col < effective_cols_b) {
            uint32_t b_base_idx;
            if (b_transposed) {
                    b_base_idx = b_col * nof_cols_b + b_row;
                } else {
                    b_base_idx = b_row * nof_cols_b + b_col;
                }
                tile_b[ty][tx] = load_field_element(mat_b, b_base_idx * degree + d);
            } else {
                tile_b[ty][tx] = 0;
            }

            __syncthreads();

            // Compute partial dot product for this tile and coefficient
            #pragma unroll 8
            for (uint32_t k = 0; k < POLY_TILE; ++k) {
                acc[d] = mod_add64(acc[d], mod_mul64(tile_a[ty][k], tile_b[k][tx]));
            }

            __syncthreads();
        }
    }

    // Write results
    if (row < effective_rows_a && col < effective_cols_b) {
        uint32_t base_out_idx = (row * effective_cols_b + col) * degree;
        for (uint32_t d = 0; d < degree; ++d) {
            store_field_element(mat_out, base_out_idx + d, acc[d]);
        }
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

        // 使用内存池分配（优化：避免频繁malloc/free）
        auto& pool = CUDAMemoryPool::get_instance();
        cudaStream_t stream = CUDAStreamPool::get_instance().get_stream();
        
        uint32_t *d_a = static_cast<uint32_t*>(pool.allocate(size_a, stream));
        uint32_t *d_b = static_cast<uint32_t*>(pool.allocate(size_b, stream));
        uint32_t *d_out = static_cast<uint32_t*>(pool.allocate(size_out, stream));

        if (!d_a || !d_b || !d_out) {
            std::cerr << "[CUSTOM MATMUL] Memory pool allocation failed" << std::endl;
            if (d_a) pool.deallocate(d_a, size_a, stream);
            if (d_b) pool.deallocate(d_b, size_b, stream);
            if (d_out) pool.deallocate(d_out, size_out, stream);
            return -1;
        }

        // 异步拷贝输入（优化：重叠传输）
        cudaMemcpyAsync(d_a, mat_a_host, size_a, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b, mat_b_host, size_b, cudaMemcpyHostToDevice, stream);

        // 配置 kernel - 使用 TILE_SIZE (32x32)
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(
            (effective_cols_b + TILE_SIZE - 1) / TILE_SIZE,
            (effective_rows_a + TILE_SIZE - 1) / TILE_SIZE
        );

        // 启动 kernel（在stream上异步执行）
        matmul_scalar_kernel<<<grid, block, 0, stream>>>(
            d_a, d_b, d_out,
            effective_rows_a, effective_cols_a, effective_cols_b,
            nof_cols_a, nof_cols_b,
            a_transposed, b_transposed
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM MATMUL] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            pool.deallocate(d_a, size_a, stream);
            pool.deallocate(d_b, size_b, stream);
            pool.deallocate(d_out, size_out, stream);
            return -1;
        }

        // 异步拷贝结果
        cudaMemcpyAsync((void*)mat_out_host, d_out, size_out, cudaMemcpyDeviceToHost, stream);
        
        // 同步等待完成
        cudaStreamSynchronize(stream);

        // 归还内存到池（优化：复用内存）
        pool.deallocate(d_a, size_a, stream);
        pool.deallocate(d_b, size_b, stream);
        pool.deallocate(d_out, size_out, stream);

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

        // 使用内存池分配
        auto& pool = CUDAMemoryPool::get_instance();
        cudaStream_t stream = CUDAStreamPool::get_instance().get_stream();
        
        uint32_t *d_a = static_cast<uint32_t*>(pool.allocate(size_a, stream));
        uint32_t *d_b = static_cast<uint32_t*>(pool.allocate(size_b, stream));
        uint32_t *d_out = static_cast<uint32_t*>(pool.allocate(size_out, stream));

        if (!d_a || !d_b || !d_out) {
            std::cerr << "[CUSTOM MATMUL POLY] Memory pool allocation failed" << std::endl;
            if (d_a) pool.deallocate(d_a, size_a, stream);
            if (d_b) pool.deallocate(d_b, size_b, stream);
            if (d_out) pool.deallocate(d_out, size_out, stream);
            return -1;
        }

        // 异步拷贝输入
        cudaMemcpyAsync(d_a, mat_a_host, size_a, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b, mat_b_host, size_b, cudaMemcpyHostToDevice, stream);

        // 配置 kernel - 使用 16x16 tile for polynomial (smaller due to register pressure)
        const int POLY_TILE_SIZE = 16;
        dim3 block(POLY_TILE_SIZE, POLY_TILE_SIZE);
        dim3 grid(
            (effective_cols_b + POLY_TILE_SIZE - 1) / POLY_TILE_SIZE,
            (effective_rows_a + POLY_TILE_SIZE - 1) / POLY_TILE_SIZE
        );

        // 启动 kernel（在stream上异步执行）
        matmul_poly_kernel<<<grid, block, 0, stream>>>(
            d_a, d_b, d_out,
            effective_rows_a, effective_cols_a, effective_cols_b,
            nof_cols_a, nof_cols_b,
            a_transposed, b_transposed,
            degree
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM MATMUL POLY] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            pool.deallocate(d_a, size_a, stream);
            pool.deallocate(d_b, size_b, stream);
            pool.deallocate(d_out, size_out, stream);
            return -1;
        }

        // 异步拷贝结果
        cudaMemcpyAsync((void*)mat_out_host, d_out, size_out, cudaMemcpyDeviceToHost, stream);
        
        // 同步等待完成
        cudaStreamSynchronize(stream);

        // 归还内存到池
        pool.deallocate(d_a, size_a, stream);
        pool.deallocate(d_b, size_b, stream);
        pool.deallocate(d_out, size_out, stream);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM MATMUL POLY] Exception: " << e.what() << std::endl;
        return -1;
    }
}
