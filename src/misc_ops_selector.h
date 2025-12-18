/*****************************************************************************
 * Miscellaneous Operations 选择器 - 动态切换实现
 * 
 * 功能：
 * - 根据全局标志 g_use_custom_misc_ops 动态选择自定义 CUDA 或 ICICLE 默认实现
 * - 提供统一接口，无需修改调用点
 * - 支持 decompose、recompose、jl_projection、matrix_transpose
 *****************************************************************************/

#pragma once

#include "labrador.h"
#include "icicle/balanced_decomposition.h"
#include "icicle/jl_projection.h"
#include "icicle/mat_ops.h"
#include "icicle/hash/keccak.h"
#include "custom_ntt_constants.h"
#include <type_traits>
#include <iomanip>

// 全局标志：是否使用自定义 Misc Operations（在 example.cpp 中定义）
extern bool g_use_custom_misc_ops;

// Misc Operations 选择器命名空间
namespace misc_ops_selector {

using namespace icicle;
using namespace icicle::labrador;

// 声明自定义 Misc Operations 的 C 接口（实现在 custom_misc_ops_cuda.cu）
extern "C" {
    int custom_decompose_cuda(
        const void* input_host,
        uint64_t input_size,
        uint32_t base,
        void* output_host,
        uint64_t output_size,
        uint32_t degree
    );

    int custom_recompose_cuda(
        const void* input_host,
        uint64_t input_size,
        uint32_t base,
        void* output_host,
        uint64_t output_size,
        uint32_t degree
    );

    int custom_jl_projection_cuda(
        const void* input_host,
        uint64_t input_size,
        const uint8_t* seed,
        uint64_t seed_len,
        void* output_host,
        uint64_t output_size
    );

    int custom_jl_projection_with_matrix_cuda(
        const void* input_host,
        uint64_t input_size,
        const int8_t* matrix_host,
        void* output_host,
        uint64_t output_size
    );

    int custom_matrix_transpose_cuda(
        const void* input_host,
        uint32_t nof_rows,
        uint32_t nof_cols,
        void* output_host,
        uint32_t degree
    );
}

// C++17 兼容的类型检测辅助模板（复用）
template<typename T, typename = void>
struct has_d_member : std::false_type {};

template<typename T>
struct has_d_member<T, std::void_t<decltype(T::d)>> : std::true_type {};

template<typename T>
constexpr uint32_t degree_of() {
    if constexpr (has_d_member<T>::value) {
        return T::d;
    } else {
        return 1;  // 标量类型
    }
}

/**
 * @brief 智能 decompose 选择器
 */
template<typename T>
inline eIcicleError smart_decompose(
    const T* input,
    size_t input_size,
    uint32_t base,
    const VecOpsConfig& config,
    T* output,
    size_t output_size
) {
    if (g_use_custom_misc_ops) {
        std::cout << "[MISC OPS SELECTOR] Using CUSTOM decompose, input_size=" << input_size 
                  << ", base=" << base << std::endl;
        
        // 暂不支持批量模式
        if (config.batch_size > 1 || config.columns_batch) {
            std::cerr << "[MISC OPS SELECTOR] Warning: batch/columns_batch not supported in custom decompose, falling back to ICICLE" << std::endl;
            return icicle::balanced_decomposition::decompose(input, input_size, base, config, output, output_size);
        }

        uint32_t degree = degree_of<T>();
        int result = custom_decompose_cuda(
            reinterpret_cast<const void*>(input),
            input_size,
            base,
            reinterpret_cast<void*>(output),
            output_size,
            degree
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            std::cerr << "[MISC OPS SELECTOR] Custom decompose failed, falling back to ICICLE" << std::endl;
            return icicle::balanced_decomposition::decompose(input, input_size, base, config, output, output_size);
        }
    } else {
        return icicle::balanced_decomposition::decompose(input, input_size, base, config, output, output_size);
    }
}

/**
 * @brief 智能 recompose 选择器
 */
template<typename T>
inline eIcicleError smart_recompose(
    const T* input,
    size_t input_size,
    uint32_t base,
    const VecOpsConfig& config,
    T* output,
    size_t output_size
) {
    if (g_use_custom_misc_ops) {
        std::cout << "[MISC OPS SELECTOR] Using CUSTOM recompose, output_size=" << output_size 
                  << ", base=" << base << std::endl;
        
        // 暂不支持批量模式
        if (config.batch_size > 1 || config.columns_batch) {
            std::cerr << "[MISC OPS SELECTOR] Warning: batch/columns_batch not supported in custom recompose, falling back to ICICLE" << std::endl;
            return icicle::balanced_decomposition::recompose(input, input_size, base, config, output, output_size);
        }

        uint32_t degree = degree_of<T>();
        int result = custom_recompose_cuda(
            reinterpret_cast<const void*>(input),
            input_size,
            base,
            reinterpret_cast<void*>(output),
            output_size,
            degree
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            std::cerr << "[MISC OPS SELECTOR] Custom recompose failed, falling back to ICICLE" << std::endl;
            return icicle::balanced_decomposition::recompose(input, input_size, base, config, output, output_size);
        }
    } else {
        return icicle::balanced_decomposition::recompose(input, input_size, base, config, output, output_size);
    }
}

/**
 * @brief 智能 jl_projection 选择器
 */
inline eIcicleError smart_jl_projection(
    const Zq* input,
    size_t input_size,
    const std::byte* seed,
    size_t seed_len,
    const VecOpsConfig& config,
    Zq* output,
    size_t output_size
) {
    if (g_use_custom_misc_ops) {
        std::cout << "[MISC OPS SELECTOR] Using CUSTOM jl_projection, " << input_size
                  << " -> " << output_size << std::endl;

        // 暂不支持批量模式
        if (config.batch_size > 1) {
            std::cerr << "[MISC OPS SELECTOR] Warning: batch not supported in custom jl_projection, falling back to ICICLE" << std::endl;
            return icicle::jl_projection(input, input_size, seed, seed_len, config, output, output_size);
        }

        // 使用 ICICLE 的 get_jl_matrix_rows 生成矩阵，确保与 ICICLE 完全一致
        std::vector<Zq> icicle_rows(output_size * input_size);
        eIcicleError rows_err = icicle::get_jl_matrix_rows(
            seed, seed_len, input_size, 0, output_size, config, icicle_rows.data());
        
        if (rows_err != eIcicleError::SUCCESS) {
            std::cerr << "[MISC OPS SELECTOR] get_jl_matrix_rows failed, falling back to ICICLE" << std::endl;
            return icicle::jl_projection(input, input_size, seed, seed_len, config, output, output_size);
        }

        // 解码 ICICLE 矩阵行为 {-1, 0, 1}
        const uint64_t q = kCustomModulusQ;
        std::vector<int8_t> matrix(output_size * input_size, 0);
        
        for (size_t r = 0; r < output_size; ++r) {
            for (size_t c = 0; c < input_size; ++c) {
                const Zq& z = icicle_rows[r * input_size + c];
                const uint32_t* limbs = reinterpret_cast<const uint32_t*>(&z);
                uint64_t v = (static_cast<uint64_t>(limbs[1]) << 32) | limbs[0];
                
                int8_t val = 0;
                if (v == 1) val = 1;
                else if (v == q - 1) val = -1;
                // else val = 0 (包括 v == 0 和其他值)
                
                matrix[r * input_size + c] = val;
            }
        }

        // 调用使用预生成矩阵的 CUDA 接口
        int result = custom_jl_projection_with_matrix_cuda(
            reinterpret_cast<const void*>(input),
            input_size,
            matrix.data(),
            reinterpret_cast<void*>(output),
            output_size
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            std::cerr << "[MISC OPS SELECTOR] Custom jl_projection failed, falling back to ICICLE" << std::endl;
            return icicle::jl_projection(input, input_size, seed, seed_len, config, output, output_size);
        }
    } else {
        return icicle::jl_projection(input, input_size, seed, seed_len, config, output, output_size);
    }
}

/**
 * @brief 智能 matrix_transpose 选择器
 */
template<typename T>
inline eIcicleError smart_matrix_transpose(
    const T* mat_in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    T* mat_out
) {
    if (g_use_custom_misc_ops) {
        std::cout << "[MISC OPS SELECTOR] Using CUSTOM matrix_transpose, " 
                  << nof_rows << "x" << nof_cols << std::endl;
        
        // 暂不支持批量模式和 in-place
        if (config.batch_size > 1 || config.columns_batch || mat_in == mat_out) {
            std::cerr << "[MISC OPS SELECTOR] Warning: batch/in-place not supported in custom transpose, falling back to ICICLE" << std::endl;
            return icicle::matrix_transpose(mat_in, nof_rows, nof_cols, config, mat_out);
        }

        uint32_t degree = degree_of<T>();
        int result = custom_matrix_transpose_cuda(
            reinterpret_cast<const void*>(mat_in),
            nof_rows,
            nof_cols,
            reinterpret_cast<void*>(mat_out),
            degree
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            std::cerr << "[MISC OPS SELECTOR] Custom matrix_transpose failed, falling back to ICICLE" << std::endl;
            return icicle::matrix_transpose(mat_in, nof_rows, nof_cols, config, mat_out);
        }
    } else {
        return icicle::matrix_transpose(mat_in, nof_rows, nof_cols, config, mat_out);
    }
}

} // namespace misc_ops_selector

// 便捷宏：在代码中使用这些宏替代 icicle 的操作
#define USE_SMART_DECOMPOSE ::misc_ops_selector::smart_decompose
#define USE_SMART_RECOMPOSE ::misc_ops_selector::smart_recompose
#define USE_SMART_JL_PROJECTION ::misc_ops_selector::smart_jl_projection
#define USE_SMART_MATRIX_TRANSPOSE ::misc_ops_selector::smart_matrix_transpose
