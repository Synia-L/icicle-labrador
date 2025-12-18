/*****************************************************************************
 * Vector Operations 选择器 - 动态切换向量操作实现
 * 
 * 功能：
 * - 根据全局标志 g_use_custom_vec_ops 动态选择使用自定义 CUDA 或 ICICLE 默认实现
 * - 提供统一的接口，无需修改 prover.cpp/verifier.cpp 中的每个调用点
 * - 支持 scalar_mul_vec、vector_add、vector_mul、vector_sum
 * 
 * 使用方法：
 * 1. 在需要使用的文件中包含此头文件：#include "vec_ops_selector.h"
 * 2. 使用 USE_SMART_SCALAR_MUL_VEC / USE_SMART_VECTOR_ADD / USE_SMART_VECTOR_MUL / USE_SMART_VECTOR_SUM 宏
 * 3. 通过命令行参数控制是否启用自定义实现
 *****************************************************************************/

#pragma once

#include "labrador.h"
#include "icicle/vec_ops.h"
#include <type_traits>

// 全局标志：是否使用自定义 Vector Operations（在 example.cpp 中定义）
extern bool g_use_custom_vec_ops;

// Vector Operations 选择器命名空间
namespace vec_ops_selector {

using namespace icicle;
using namespace icicle::labrador;

// 声明自定义 Vector Operations 的 C 接口（实现在 custom_vec_ops_cuda.cu）
extern "C" {
    int custom_scalar_mul_vec_cuda(
        const void* scalar_host,
        const void* vec_host,
        void* output_host,
        uint64_t size,
        uint64_t stride,
        uint32_t degree
    );

    int custom_vector_add_cuda(
        const void* vec_a_host,
        const void* vec_b_host,
        void* output_host,
        uint64_t size,
        uint32_t degree
    );

    int custom_vector_mul_cuda(
        const void* vec_a_host,
        const void* vec_b_host,
        void* output_host,
        uint64_t size,
        uint32_t degree
    );

    int custom_vector_sum_cuda(
        const void* vec_host,
        void* output_host,
        uint64_t size,
        uint64_t stride,
        uint32_t degree
    );
}

// C++17 兼容的类型检测辅助模板
template<typename T, typename = void>
struct has_d_member : std::false_type {};

template<typename T>
struct has_d_member<T, std::void_t<decltype(T::d)>> : std::true_type {};

// 安全获取 degree（Zq -> 1，PolyRing -> d）
template<typename T, typename std::enable_if<has_d_member<T>::value, int>::type = 0>
constexpr uint32_t degree_of() {
  return static_cast<uint32_t>(T::d);
}

template<typename T, typename std::enable_if<!has_d_member<T>::value, int>::type = 0>
constexpr uint32_t degree_of() {
  return 1;
}

/**
 * @brief 智能 scalar_mul_vec 选择器
 */
template<typename T>
inline eIcicleError smart_scalar_mul_vec(
    const T* scalar,
    const T* vec,
    uint64_t size,
    const VecOpsConfig& config,
    T* output
) {
    if (g_use_custom_vec_ops) {
        std::cout << "[VEC OPS SELECTOR] Using CUSTOM scalar_mul_vec, size=" << size << std::endl;
        
        // 暂不支持批量模式（需要检查 config）
        if (config.batch_size > 1 || config.columns_batch) {
            std::cerr << "[VEC OPS SELECTOR] Warning: batch mode not supported in custom impl, falling back to ICICLE" << std::endl;
            return icicle::scalar_mul_vec(scalar, vec, size, config, output);
        }

        uint64_t stride = 1; // 默认 stride
        uint32_t degree = degree_of<T>();
        int result = custom_scalar_mul_vec_cuda(
            reinterpret_cast<const void*>(scalar),
            reinterpret_cast<const void*>(vec),
            reinterpret_cast<void*>(output),
            size,
            stride,
            degree
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            std::cerr << "[VEC OPS SELECTOR] Custom scalar_mul_vec failed, falling back to ICICLE" << std::endl;
            return icicle::scalar_mul_vec(scalar, vec, size, config, output);
        }
    } else {
        return icicle::scalar_mul_vec(scalar, vec, size, config, output);
    }
}

/**
 * @brief 智能 vector_add 选择器
 */
template<typename T>
inline eIcicleError smart_vector_add(
    const T* vec_a,
    const T* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    T* output
) {
    if (g_use_custom_vec_ops) {
        std::cout << "[VEC OPS SELECTOR] Using CUSTOM vector_add, size=" << size << std::endl;
        
        // 暂不支持批量模式
        if (config.batch_size > 1 || config.columns_batch) {
            std::cerr << "[VEC OPS SELECTOR] Warning: batch mode not supported in custom impl, falling back to ICICLE" << std::endl;
            return icicle::vector_add(vec_a, vec_b, size, config, output);
        }

        uint32_t degree = degree_of<T>();
        int result = custom_vector_add_cuda(
            reinterpret_cast<const void*>(vec_a),
            reinterpret_cast<const void*>(vec_b),
            reinterpret_cast<void*>(output),
            size,
            degree
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            std::cerr << "[VEC OPS SELECTOR] Custom vector_add failed, falling back to ICICLE" << std::endl;
            return icicle::vector_add(vec_a, vec_b, size, config, output);
        }
    } else {
        return icicle::vector_add(vec_a, vec_b, size, config, output);
    }
}

/**
 * @brief 智能 vector_mul 选择器（逐元素相乘）
 */
template<typename T>
inline eIcicleError smart_vector_mul(
    const T* vec_a,
    const T* vec_b,
    uint64_t size,
    const VecOpsConfig& config,
    T* output
) {
    if (g_use_custom_vec_ops) {
        std::cout << "[VEC OPS SELECTOR] Using CUSTOM vector_mul, size=" << size << std::endl;
        
        // 暂不支持批量模式
        if (config.batch_size > 1 || config.columns_batch) {
            std::cerr << "[VEC OPS SELECTOR] Warning: batch mode not supported in custom impl, falling back to ICICLE" << std::endl;
            return icicle::vector_mul(vec_a, vec_b, size, config, output);
        }

        uint32_t degree = degree_of<T>();
        int result = custom_vector_mul_cuda(
            reinterpret_cast<const void*>(vec_a),
            reinterpret_cast<const void*>(vec_b),
            reinterpret_cast<void*>(output),
            size,
            degree
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            std::cerr << "[VEC OPS SELECTOR] Custom vector_mul failed, falling back to ICICLE" << std::endl;
            return icicle::vector_mul(vec_a, vec_b, size, config, output);
        }
    } else {
        return icicle::vector_mul(vec_a, vec_b, size, config, output);
    }
}

/**
 * @brief 智能 vector_sum 选择器
 */
template<typename T>
inline eIcicleError smart_vector_sum(
    const T* vec,
    uint64_t size,
    const VecOpsConfig& config,
    T* output
) {
    if (g_use_custom_vec_ops) {
        std::cout << "[VEC OPS SELECTOR] Using CUSTOM vector_sum, size=" << size << std::endl;
        
        // 暂不支持批量模式
        if (config.batch_size > 1 || config.columns_batch) {
            std::cerr << "[VEC OPS SELECTOR] Warning: batch mode not supported in custom impl, falling back to ICICLE" << std::endl;
            return icicle::vector_sum(vec, size, config, output);
        }

        uint64_t stride = 1; // 默认 stride
        uint32_t degree = degree_of<T>();
        int result = custom_vector_sum_cuda(
            reinterpret_cast<const void*>(vec),
            reinterpret_cast<void*>(output),
            size,
            stride,
            degree
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            std::cerr << "[VEC OPS SELECTOR] Custom vector_sum failed, falling back to ICICLE" << std::endl;
            return icicle::vector_sum(vec, size, config, output);
        }
    } else {
        return icicle::vector_sum(vec, size, config, output);
    }
}

} // namespace vec_ops_selector

// 便捷宏：在代码中使用这些宏替代 icicle 的向量操作
#define USE_SMART_SCALAR_MUL_VEC ::vec_ops_selector::smart_scalar_mul_vec
#define USE_SMART_VECTOR_ADD ::vec_ops_selector::smart_vector_add
#define USE_SMART_VECTOR_MUL ::vec_ops_selector::smart_vector_mul
#define USE_SMART_VECTOR_SUM ::vec_ops_selector::smart_vector_sum
