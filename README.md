# Labrador with Custom CUDA NTT


> **⚠️ WARNING**: This code has not been audited. Use at your own risk.

This repository contains a compact, end-to-end demo of **LaBRADOR** — the first practical _lattice-based_ zk-SNARK (CRYPTO 2023) - built on top of **ICICLE v4**. LaBRADOR produces ~50 kB proofs without a trusted setup and is secure under the Module-SIS assumption, making it resistant to both classical and _quantum_ attacks.

ICICLE ships highly-tuned GPU and CPU kernels for FFT/NTT, polynomial arithmetic and lattice primitives. Thanks to those kernels the prover can run unchanged on a laptop CPU _or_ a CUDA-capable GPU and enjoy order-of-magnitude speed-ups.

## 项目说明

本项目在 Ingonyama 的 fast-labrador-prover 基础上，实现了自定义的 CUDA NTT，替换了 ICICLE 的默认 NTT 实现。

## 验证状态

- **Roundtrip 测试**: ✅ 通过（256/256 系数完美恢复）
- **Labrador 验证**: ✅ 通过（SUCCESS!）

## 🚀 快速开始

### 构建并运行（使用 Custom NTT + CUDA）
```bash
./run.sh -d CUDA -c
```

## 核心文件

- **`src/custom_ntt_hardcoded.cu`**: Custom CUDA NTT 实现（Radix-2 DIT/DIF，硬编码 twiddle/coset）
- **`src/ntt_selector.h`**: NTT 选择器（ICICLE vs Custom）
- **`src/example.cpp`**: Labrador 主程序
- **`src/prover.cpp`**: Prover 实现（使用 Custom NTT）
- **`src/verifier.cpp`**: Verifier 实现

## 技术特性

- **算法**: Radix-2 Decimation-In-Time (Forward) / Decimation-In-Frequency (Inverse)
- **Negacyclic NTT**: 使用 ψ (128th root of unity) 作为 coset generator
- **RNS 系统**: BabyBear (2^31 - 2^27 + 1) + KoalaBear (2^31 - 2^24 + 1)
- **多项式大小**: 64 系数
- **Batch 处理**: 支持多个多项式同时变换

## 性能

- **Proof 生成**: ~1.2 秒（n=64, r=8）
- **验证**: 通过

## 命令行选项

```bash
# 使用 ICICLE 默认 NTT (GPU)
./build/src/example -d CUDA

# 使用 Custom NTT
./build/src/example -d CUDA --custom-ntt

# CPU 模式
./build/src/example -d CPU
```

## 原理说明

Custom NTT 实现了标准的 FFT 算法，针对有限域进行了优化：

1. **Forward NTT**: 
   - 输入自然顺序 → Coset multiplication → Bit-reversal → DIT butterfly → 输出自然顺序

2. **Inverse NTT**:
   - 输入 bit-reversed → DIF butterfly → Divide by N → Inverse coset multiplication → 输出自然顺序

3. **Negacyclic 转换**:
   - 通过 coset multiplication 将 cyclic NTT 转换为 negacyclic NTT
   - 等价于在 ψ 的奇数次幂上求值



## 🙏 Credits


```cpp
// SHOW_STEPS creates a print output listing every step performed by the Prover and the time taken
constexpr bool SHOW_STEPS = true;
```

All functions and objects are documented in code.

## Performance

![LaBRADOR latency vs. constraint count](labrador-latency.png)

