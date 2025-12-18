/*****************************************************************************
 * NTT 选择 - 动态切换 NTT 实现
 * 
 * 功能：
 * - 根据全局标志 g_use_custom_ntt 动态选择使用自定义 NTT 或 ICICLE 默认 NTT
 * - 提供统一的接口，无需修改 prover.cpp 中的每个 NTT 调用点
 * 
 * 使用方法：
 * 1. 在 prover.cpp 中包含此头文件：#include "ntt_selector.h"
 * 2. 使用 USE_SMART_NTT 宏替代 icicle::labrador::ntt
 * 3. 通过命令行参数 --custom-ntt 控制是否启用自定义实现
 *****************************************************************************/

#pragma once

#include "labrador.h"

// 全局标志：是否使用自定义 NTT（在 example.cpp 中定义）
extern bool g_use_custom_ntt;

// NTT 选择器命名空间
namespace ntt_selector {

using namespace icicle::labrador;

// 声明自定义 NTT 的 C 接口（实现在 custom_ntt_radix8.cu）
extern "C" {
    void custom_ntt_init_cuda_with_roots(const uint32_t* psi_limbs, const uint32_t* omega_limbs);
    int custom_ntt_forward_cuda(void* input, void* output, int batch_size);
    int custom_ntt_inverse_cuda(void* input, void* output, int batch_size);
    void custom_ntt_cleanup_cuda();
}

/**
 * @brief 智能 NTT 选择器函数
 * 根据全局标志自动选择使用自定义 NTT 或 ICICLE 默认实现
 * 
 * @param input 输入多项式数组
 * @param size batch 大小（多项式数量）
 * @param dir NTT 方向（Forward/Inverse）
 * @param config NTT 配置
 * @param output 输出多项式数组
 * @return ICICLE 错误码
 */
inline eIcicleError smart_ntt(
    const PolyRing* input,
    size_t size,
    NTTDir dir,
    const NegacyclicNTTConfig& config,
    PolyRing* output
) {
    if (g_use_custom_ntt) {
        // 使用自定义 CUDA NTT 实现 (Forward + Inverse)
        std::cout << "[NTT SELECTOR] Using CUSTOM NTT (" 
                  << (dir == NTTDir::kForward ? "Forward" : "Inverse") 
                  << "), " << size << " polys" << std::endl;
        
        int result;
        if (dir == NTTDir::kForward) {
            result = custom_ntt_forward_cuda((void*)input, (void*)output, size);
        } else {
            result = custom_ntt_inverse_cuda((void*)input, (void*)output, size);
        }
        
        if (result == 0) {
            return eIcicleError::SUCCESS;
        } else {
            // 失败时回退到 ICICLE 实现
            std::cerr << "[NTT SELECTOR] Custom NTT failed, falling back to ICICLE" << std::endl;
            return icicle::labrador::ntt(input, size, dir, config, output);
        }
    } else {
        // 使用默认 ICICLE 实现（CPU 或 GPU）
        return icicle::labrador::ntt(input, size, dir, config, output);
    }
}

} // namespace ntt_selector

// 便捷宏：在代码中使用此宏替代 icicle::labrador::ntt
#define USE_SMART_NTT ::ntt_selector::smart_ntt

