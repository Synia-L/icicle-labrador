/*****************************************************************************
 * MatMul 选择器 - 动态切换 MatMul 实现
 * 
 * 功能：
 * - 根据全局标志 g_use_custom_matmul 动态选择使用自定义 CUDA MatMul 或 ICICLE 默认 MatMul
 * - 提供统一的接口，无需修改 prover.cpp/verifier.cpp 中的每个 matmul 调用点
 * - 支持标量域和多项式环版本
 * 
 * 使用方法：
 * 1. 在需要使用的文件中包含此头文件：#include "matmul_selector.h"
 * 2. 使用 USE_SMART_MATMUL 宏替代 icicle::matmul
 * 3. 通过命令行参数控制是否启用自定义实现
 *****************************************************************************/

#pragma once

#include "labrador.h"
#include "icicle/mat_ops.h"
#include <type_traits>

// 全局标志：是否使用自定义 MatMul（在 example.cpp 中定义）
extern bool g_use_custom_matmul;

// MatMul 选择器命名空间
namespace matmul_selector {

using namespace icicle;
using namespace icicle::labrador;

// 声明自定义 MatMul 的 C 接口（实现在 custom_matmul_cuda.cu）
extern "C" {
    int custom_matmul_scalar_cuda(
        const void* mat_a_host,
        uint32_t nof_rows_a,
        uint32_t nof_cols_a,
        const void* mat_b_host,
        uint32_t nof_rows_b,
        uint32_t nof_cols_b,
        void* mat_out_host,
        bool a_transposed,
        bool b_transposed
    );

    int custom_matmul_poly_cuda(
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
    );
}

/**
 * @brief 智能 MatMul 选择器函数 - 标量域版本
 * 
 * @tparam T 标量类型（如 Zq）
 */
template<typename T>
inline eIcicleError smart_matmul_scalar(
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    T* mat_out
) {
    if (g_use_custom_matmul) {
        std::cout << "[MATMUL SELECTOR] Using CUSTOM MatMul (Scalar) "
                  << nof_rows_a << "x" << nof_cols_a << " @ " 
                  << nof_rows_b << "x" << nof_cols_b << std::endl;
        
        // 检查不支持的配置
        if (config.result_transposed) {
            std::cerr << "[MATMUL SELECTOR] Warning: result_transposed not supported in custom impl, falling back to ICICLE" << std::endl;
            return icicle::matmul(mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
        }

        int result = custom_matmul_scalar_cuda(
            reinterpret_cast<const void*>(mat_a),
            nof_rows_a,
            nof_cols_a,
            reinterpret_cast<const void*>(mat_b),
            nof_rows_b,
            nof_cols_b,
            reinterpret_cast<void*>(mat_out),
            config.a_transposed,
            config.b_transposed
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            // 失败时回退到 ICICLE 实现
            std::cerr << "[MATMUL SELECTOR] Custom MatMul failed, falling back to ICICLE" << std::endl;
            return icicle::matmul(mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
        }
    } else {
        // 使用默认 ICICLE 实现（CPU 或 GPU）
        return icicle::matmul(mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
    }
}

/**
 * @brief 智能 MatMul 选择器函数 - 多项式环版本
 * 
 * @tparam T 多项式环类型（如 PolyRing）
 */
template<typename T>
inline eIcicleError smart_matmul_poly(
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    T* mat_out
) {
    if (g_use_custom_matmul) {
        constexpr uint32_t degree = T::d;  // 多项式度数
        
        std::cout << "[MATMUL SELECTOR] Using CUSTOM MatMul (Poly, degree=" << degree << ") "
                  << nof_rows_a << "x" << nof_cols_a << " @ " 
                  << nof_rows_b << "x" << nof_cols_b << std::endl;

        // 检查不支持的配置
        if (config.result_transposed) {
            std::cerr << "[MATMUL SELECTOR] Warning: result_transposed not supported in custom impl, falling back to ICICLE" << std::endl;
            return icicle::matmul(mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
        }

        int result = custom_matmul_poly_cuda(
            reinterpret_cast<const void*>(mat_a),
            nof_rows_a,
            nof_cols_a,
            reinterpret_cast<const void*>(mat_b),
            nof_rows_b,
            nof_cols_b,
            reinterpret_cast<void*>(mat_out),
            config.a_transposed,
            config.b_transposed,
            degree
        );

        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            // 失败时回退到 ICICLE 实现
            std::cerr << "[MATMUL SELECTOR] Custom MatMul (Poly) failed, falling back to ICICLE" << std::endl;
            return icicle::matmul(mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
        }
    } else {
        // 使用默认 ICICLE 实现（CPU 或 GPU）
        return icicle::matmul(mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
    }
}

// C++17 兼容的类型检测辅助模板
template<typename T, typename = void>
struct has_d_member : std::false_type {};

template<typename T>
struct has_d_member<T, std::void_t<decltype(T::d)>> : std::true_type {};

/**
 * @brief 通用智能 MatMul 选择器 - 自动识别标量 vs 多项式环（多项式环版本）
 */
template<typename T>
inline typename std::enable_if<has_d_member<T>::value && (T::d > 1), eIcicleError>::type
smart_matmul(
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    T* mat_out
) {
    return smart_matmul_poly(mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
}

/**
 * @brief 通用智能 MatMul 选择器 - 自动识别标量 vs 多项式环（标量版本）
 */
template<typename T>
inline typename std::enable_if<!has_d_member<T>::value || (T::d <= 1), eIcicleError>::type
smart_matmul(
    const T* mat_a,
    uint32_t nof_rows_a,
    uint32_t nof_cols_a,
    const T* mat_b,
    uint32_t nof_rows_b,
    uint32_t nof_cols_b,
    const MatMulConfig& config,
    T* mat_out
) {
    return smart_matmul_scalar(mat_a, nof_rows_a, nof_cols_a, mat_b, nof_rows_b, nof_cols_b, config, mat_out);
}

} // namespace matmul_selector

// 便捷宏：在代码中使用此宏替代 icicle::matmul
#define USE_SMART_MATMUL ::matmul_selector::smart_matmul
