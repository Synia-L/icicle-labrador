/*****************************************************************************
 * Custom CUDA Miscellaneous Operations Implementation
 * 
 * 实现三个操作：
 * 1. decompose/recompose - 平衡分解/重组
 * 2. jl_projection - Johnson-Lindenstrauss 投影  
 * 3. matrix_transpose - 矩阵转置
 * 
 * 参考 CPU 版本逻辑
 *****************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cstring>
#include "custom_ntt_constants.h"
#include "icicle/hash/keccak.h"
#include "icicle/hash/hash.h"

// OpenMP for parallel matrix generation
#ifdef _OPENMP
#include <omp.h>
#endif

// ============ 常量定义 ============
const int LIMBS_COUNT_MISC = kCustomLimbs;
const uint64_t MODULUS_Q_MISC = kCustomModulusQ;

// ============ 设备端辅助函数 ============

__host__ __device__ __forceinline__ uint64_t combine_u64_misc(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

__host__ __device__ __forceinline__ void split_u64_misc(uint64_t value, uint32_t& lo, uint32_t& hi) {
    lo = static_cast<uint32_t>(value & 0xffffffffULL);
    hi = static_cast<uint32_t>(value >> 32);
}

__device__ __forceinline__ uint64_t load_element_misc(const uint32_t* data, uint64_t idx) {
    return combine_u64_misc(data[idx * LIMBS_COUNT_MISC], data[idx * LIMBS_COUNT_MISC + 1]);
}

__device__ __forceinline__ void store_element_misc(uint32_t* data, uint64_t idx, uint64_t value) {
    uint32_t lo, hi;
    split_u64_misc(value, lo, hi);
    data[idx * LIMBS_COUNT_MISC] = lo;
    data[idx * LIMBS_COUNT_MISC + 1] = hi;
}

__host__ __device__ __forceinline__ uint64_t mod_add64_misc(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    if (sum >= MODULUS_Q_MISC || sum < a) sum -= MODULUS_Q_MISC;
    return sum;
}

__host__ __device__ __forceinline__ uint64_t mod_sub64_misc(uint64_t a, uint64_t b) {
    if (a >= b) return a - b;
    return a + (MODULUS_Q_MISC - b);
}

__host__ __device__ __forceinline__ uint64_t mod_mul64_misc(uint64_t a, uint64_t b) {
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return static_cast<uint64_t>(prod % MODULUS_Q_MISC);
}

// ============ 1. Balanced Decomposition CUDA Kernels ============

/**
 * @brief 设备端 divmod 函数 - Python 风格的除法和取模
 * 保证商向下取整，余数非负
 */
__device__ __forceinline__ void divmod_device(int64_t a, int64_t base, int64_t& quotient, int64_t& remainder) {
    quotient = a / base;
    remainder = a % base;
    
    // 如果余数非零且 a 和 base 符号不同，需要调整
    if ((remainder != 0) && ((a ^ base) < 0)) {
        quotient -= 1;
        remainder += base;
    }
}

/**
 * @brief 平衡分解内核
 * 将 Zq 元素分解为平衡的 base-b digits
 */
__global__ void decompose_balanced_kernel(
    const uint32_t* input,
    uint32_t* output,
    uint64_t input_size,
    uint32_t base,
    uint32_t digits_per_element,
    uint32_t degree
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_size) return;

    const int64_t q = MODULUS_Q_MISC;
    const int64_t base_i64 = static_cast<int64_t>(base);
    const int64_t base_div2 = base / 2;
    const int64_t q_div2 = q / 2;

    for (uint32_t d = 0; d < degree; ++d) {
        int64_t val = static_cast<int64_t>(load_element_misc(input, idx * degree + d));
        
        // 处理 val > q/2 的情况（转为负数表示）
        if (base > 2 && val > q_div2) {
            val -= q;
        }

        // 逐digit分解
        for (uint32_t digit_idx = 0; digit_idx < digits_per_element; ++digit_idx) {
            int64_t digit;
            divmod_device(val, base_i64, val, digit);
            
            // 调整为平衡范围 [-b/2, b/2)
            if (digit > base_div2) {
                digit -= base;
                val++;
            }
            
            // 转换为正数存储
            uint64_t digit_unsigned = (digit < 0) ? (digit + q) : digit;
            
            // digit-major 布局：output[digit_idx][idx]
            store_element_misc(output, digit_idx * input_size * degree + idx * degree + d, digit_unsigned);
        }
    }
}

/**
 * @brief 平衡重组内核
 * 从平衡的 base-b digits 重组为 Zq 元素
 */
__global__ void recompose_balanced_kernel(
    const uint32_t* input,
    uint32_t* output,
    uint64_t output_size,
    uint32_t base,
    uint32_t digits_per_element,
    uint32_t degree
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    for (uint32_t d = 0; d < degree; ++d) {
        uint64_t acc = 0;
        
        // 从最高位 digit 开始累加（Horner 方法）
        // acc = digit[n-1] * base^(n-1) + ... + digit[1] * base + digit[0]
        //     = ((...((digit[n-1]) * base + digit[n-2]) * base + ...) * base + digit[0]
        for (int digit_idx = digits_per_element - 1; digit_idx >= 0; --digit_idx) {
            uint64_t digit = load_element_misc(input, digit_idx * output_size * degree + idx * degree + d);
            acc = mod_mul64_misc(acc, base);
            acc = mod_add64_misc(acc, digit);
        }
        
        store_element_misc(output, idx * degree + d, acc);
    }
}

// ============ 2. Keccak512 Device Implementation ============

// Keccak512 常量（从 ICICLE CPU 实现移植）
#define KECCAK_ROUNDS 24
#define SHA3_KECCAK_SPONGE_WORDS 25  // 1600 bits / 64 bits
#define SHA3_USE_KECCAK_FLAG 0x80000000
#define SHA3_ROTL64(x, y) (((x) << (y)) | ((x) >> ((sizeof(uint64_t) * 8) - (y))))

__device__ const uint64_t keccakf_rndc[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

__device__ const unsigned keccakf_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ const unsigned keccakf_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

/**
 * @brief Keccak-f 轮函数（设备端）
 * 对 25 个 uint64_t 的状态进行 24 轮置换
 */
__device__ void keccakf_device(uint64_t s[25]) {
    int i, j, round;
    uint64_t t, bc[5];

    for (round = 0; round < KECCAK_ROUNDS; round++) {
        /* Theta */
        for (i = 0; i < 5; i++)
            bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ SHA3_ROTL64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                s[j + i] ^= t;
        }

        /* Rho Pi */
        t = s[1];
        for (i = 0; i < 24; i++) {
            j = keccakf_piln[i];
            bc[0] = s[j];
            s[j] = SHA3_ROTL64(t, keccakf_rotc[i]);
            t = bc[0];
        }

        /* Chi */
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++)
                bc[i] = s[j + i];
            for (i = 0; i < 5; i++)
                s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        /* Iota */
        s[0] ^= keccakf_rndc[round];
    }
}

/**
 * @brief Keccak512 哈希函数（设备端）
 * 输入：input（字节数组），input_len（字节长度）
 * 输出：output（64 字节）
 */
__device__ void keccak512_hash_device(const uint8_t* input, uint32_t input_len, uint8_t* output) {
    uint64_t s[SHA3_KECCAK_SPONGE_WORDS] = {0};
    const uint32_t capacityWords = 2 * 512 / (8 * sizeof(uint64_t));  // 16 words for 512-bit
    const uint32_t rateWords = SHA3_KECCAK_SPONGE_WORDS - capacityWords;  // 9 words

    // 吸收阶段：将输入异或到状态中（与CPU实现完全一致）
    uint32_t wordIndex = 0;
    uint32_t byteIndex = 0;
    uint64_t saved = 0;

    // 处理输入字节（逐字节处理，累积到saved中）
    for (uint32_t i = 0; i < input_len; i++) {
        saved |= ((uint64_t)input[i]) << (byteIndex * 8);
        byteIndex++;

        if (byteIndex == 8) {
            // 完整的word，异或到状态中
            s[wordIndex] ^= saved;
            saved = 0;
            byteIndex = 0;
            wordIndex++;
            if (wordIndex == rateWords) {
                // 状态已满，应用Keccak-f
                keccakf_device(s);
                wordIndex = 0;
            }
        }
    }

    // 填充：Keccak 使用 0x01 || 0x00* || 0x80
    // 与CPU实现完全一致：t = 1 << (byteIndex * 8)
    uint64_t t = ((uint64_t)1) << (byteIndex * 8);
    s[wordIndex] ^= saved ^ t;
    // 在最后一个rate word的最高位设置0x80
    // CPU实现：SHA3_KECCAK_SPONGE_WORDS - SHA3_CW(capacityWords) - 1 = 25 - 16 - 1 = 8
    s[rateWords - 1] ^= 0x8000000000000000UL;

    // 挤压阶段：应用 Keccak-f
    keccakf_device(s);

    // 提取前 64 字节（512 位）
    // 注意：CPU实现使用小端序，直接提取字节
    for (uint32_t i = 0; i < 8; i++) {  // 8 words = 64 bytes
        uint64_t word = s[i];
        // 小端序：低字节在前
        output[i * 8 + 0] = (uint8_t)(word & 0xff);
        output[i * 8 + 1] = (uint8_t)((word >> 8) & 0xff);
        output[i * 8 + 2] = (uint8_t)((word >> 16) & 0xff);
        output[i * 8 + 3] = (uint8_t)((word >> 24) & 0xff);
        output[i * 8 + 4] = (uint8_t)((word >> 32) & 0xff);
        output[i * 8 + 5] = (uint8_t)((word >> 40) & 0xff);
        output[i * 8 + 6] = (uint8_t)((word >> 48) & 0xff);
        output[i * 8 + 7] = (uint8_t)((word >> 56) & 0xff);
    }
}

// ============ 3. JL Projection CUDA Kernel ============

/**
 * @brief JL 投影内核 - 方案B：使用预生成的矩阵
 * 矩阵在主机端使用ICICLE的Keccak512生成，确保正确性
 * 
 * output[i] = sum_j matrix[i][j] * input[j]
 * matrix 是稀疏的 {-1, 0, +1} 矩阵
 */
__global__ void jl_projection_kernel(
    const uint32_t* input,
    const int8_t* matrix,  // 预生成的 {-1, 0, +1} 矩阵，row-major: matrix[row * input_size + col]
    uint32_t* output,
    uint64_t input_size,
    uint64_t output_size
) {
    uint64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= output_size) return;

    uint64_t acc = 0;
    uint64_t add_count = 0;
    uint64_t sub_count = 0;
    
    // 遍历该行的所有列
    for (uint64_t col_idx = 0; col_idx < input_size; ++col_idx) {
        int8_t matrix_val = matrix[row_idx * input_size + col_idx];
        if (matrix_val == 0) continue;  // 跳过0值，减少计算
        
        // 注意：Zq类型在内存中是连续的storage<2>结构体
        // 每个Zq元素占用2个uint32_t，所以索引应该是col_idx * 2
        uint64_t input_val = load_element_misc(input, col_idx);
        
        if (matrix_val == 1) {
            acc = mod_add64_misc(acc, input_val);
            add_count++;
        } else { // matrix_val == -1
            acc = mod_sub64_misc(acc, input_val);
            sub_count++;
        }
    }
    
    // 调试：只对前2行输出统计信息
    if (row_idx < 2) {
        // 使用printf需要特殊处理，这里先不输出
    }
    
    store_element_misc(output, row_idx, acc);
}

// ============ 3. Matrix Transpose CUDA Kernel ============

/**
 * @brief 矩阵转置内核（out-of-place）
 * 
 * output[j * nof_rows + i] = input[i * nof_cols + j]
 * 
 * 每个线程处理一个元素的所有 degree 系数
 */
__global__ void matrix_transpose_kernel(
    const uint32_t* input,
    uint32_t* output,
    uint32_t nof_rows,
    uint32_t nof_cols,
    uint32_t degree
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= nof_rows || col >= nof_cols) return;
    
    // 转置：input[row][col] -> output[col][row]
    for (uint32_t d = 0; d < degree; ++d) {
        uint64_t in_idx = (row * nof_cols + col) * degree + d;
        uint64_t out_idx = (col * nof_rows + row) * degree + d;
        
        uint64_t val = load_element_misc(input, in_idx);
        store_element_misc(output, out_idx, val);
    }
}

// ============ 主机端接口 ============

/**
 * @brief 平衡分解主机接口
 */
extern "C" int custom_decompose_cuda(
    const void* input_host,
    uint64_t input_size,
    uint32_t base,
    void* output_host,
    uint64_t output_size,
    uint32_t degree
) {
    try {
        uint32_t digits_per_element = output_size / input_size;
        if (output_size % input_size != 0) {
            std::cerr << "[CUSTOM DECOMPOSE] Error: output_size must divide input_size" << std::endl;
            return -1;
        }

        size_t input_bytes = input_size * degree * LIMBS_COUNT_MISC * sizeof(uint32_t);
        size_t output_bytes = output_size * degree * LIMBS_COUNT_MISC * sizeof(uint32_t);

        uint32_t *d_input, *d_output;
        cudaMalloc(&d_input, input_bytes);
        cudaMalloc(&d_output, output_bytes);

        cudaMemcpy(d_input, input_host, input_bytes, cudaMemcpyHostToDevice);

        int block_size = 256;
        int grid_size = (input_size + block_size - 1) / block_size;

        decompose_balanced_kernel<<<grid_size, block_size>>>(
            d_input, d_output, input_size, base, digits_per_element, degree
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM DECOMPOSE] Kernel failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_input); cudaFree(d_output);
            return -1;
        }

        cudaDeviceSynchronize();
        cudaMemcpy((void*)output_host, d_output, output_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM DECOMPOSE] Exception: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief 平衡重组主机接口
 */
extern "C" int custom_recompose_cuda(
    const void* input_host,
    uint64_t input_size,
    uint32_t base,
    void* output_host,
    uint64_t output_size,
    uint32_t degree
) {
    try {
        uint32_t digits_per_element = input_size / output_size;
        if (input_size % output_size != 0) {
            std::cerr << "[CUSTOM RECOMPOSE] Error: input_size must divide output_size" << std::endl;
            return -1;
        }

        size_t input_bytes = input_size * degree * LIMBS_COUNT_MISC * sizeof(uint32_t);
        size_t output_bytes = output_size * degree * LIMBS_COUNT_MISC * sizeof(uint32_t);

        uint32_t *d_input, *d_output;
        cudaMalloc(&d_input, input_bytes);
        cudaMalloc(&d_output, output_bytes);

        cudaMemcpy(d_input, input_host, input_bytes, cudaMemcpyHostToDevice);

        int block_size = 256;
        int grid_size = (output_size + block_size - 1) / block_size;

        recompose_balanced_kernel<<<grid_size, block_size>>>(
            d_input, d_output, output_size, base, digits_per_element, degree
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM RECOMPOSE] Kernel failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_input); cudaFree(d_output);
            return -1;
        }

        cudaDeviceSynchronize();
        cudaMemcpy((void*)output_host, d_output, output_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM RECOMPOSE] Exception: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief JL 投影主机接口（已废弃，使用 custom_jl_projection_with_matrix_cuda）
 * 
 * 此接口不再使用，保留作为参考
 */
extern "C" int custom_jl_projection_cuda(
    const void* input_host,
    uint64_t input_size,
    const uint8_t* seed,
    uint64_t seed_len,
    void* output_host,
    uint64_t output_size
) {
    std::cerr << "[CUSTOM JL_PROJECTION] This interface is deprecated, use custom_jl_projection_with_matrix_cuda instead" << std::endl;
    return -1;
}

/**
 * @brief JL 投影主机接口（使用预生成矩阵）
 * @param matrix_host 预生成的 {-1,0,1} 矩阵（row-major, size = output_size * input_size）
 */
extern "C" int custom_jl_projection_with_matrix_cuda(
    const void* input_host,
    uint64_t input_size,
    const int8_t* matrix_host,
    void* output_host,
    uint64_t output_size
) {
    try {
        size_t input_bytes = input_size * LIMBS_COUNT_MISC * sizeof(uint32_t);
        size_t output_bytes = output_size * LIMBS_COUNT_MISC * sizeof(uint32_t);
        size_t matrix_bytes = output_size * input_size * sizeof(int8_t);

        uint32_t *d_input, *d_output;
        int8_t *d_matrix;
        cudaMalloc(&d_input, input_bytes);
        cudaMalloc(&d_output, output_bytes);
        cudaMalloc(&d_matrix, matrix_bytes);

        cudaMemcpy(d_input, input_host, input_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix, matrix_host, matrix_bytes, cudaMemcpyHostToDevice);

        int block_size = 256;
        int grid_size = (output_size + block_size - 1) / block_size;
        jl_projection_kernel<<<grid_size, block_size>>>(d_input, d_matrix, d_output, input_size, output_size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM JL_PROJECTION] with_matrix kernel failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_input); cudaFree(d_output); cudaFree(d_matrix);
            return -1;
        }

        cudaDeviceSynchronize();
        cudaMemcpy((void*)output_host, d_output, output_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_matrix);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM JL_PROJECTION] with_matrix exception: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief 矩阵转置主机接口
 */
extern "C" int custom_matrix_transpose_cuda(
    const void* input_host,
    uint32_t nof_rows,
    uint32_t nof_cols,
    void* output_host,
    uint32_t degree
) {
    try {
        size_t mat_bytes = nof_rows * nof_cols * degree * LIMBS_COUNT_MISC * sizeof(uint32_t);

        uint32_t *d_input, *d_output;
        cudaMalloc(&d_input, mat_bytes);
        cudaMalloc(&d_output, mat_bytes);

        cudaMemcpy(d_input, input_host, mat_bytes, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((nof_cols + 15) / 16, (nof_rows + 15) / 16);

        matrix_transpose_kernel<<<grid, block>>>(
            d_input, d_output, nof_rows, nof_cols, degree
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUSTOM MATRIX_TRANSPOSE] Kernel failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_input); cudaFree(d_output);
            return -1;
        }

        cudaDeviceSynchronize();
        cudaMemcpy((void*)output_host, d_output, mat_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CUSTOM MATRIX_TRANSPOSE] Exception: " << e.what() << std::endl;
        return -1;
    }
}
