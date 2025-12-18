# Custom CUDA for LaBRADOR


## 使用方式

```bash
# 只使用 ICICLE 默认 CUDA 后端
./run.sh -d CUDA

# 启用自定义 NTT
./run.sh -d CUDA -c

# 启用自定义 MatMul
./run.sh -d CUDA -m

# 启用自定义 VecOps
./run.sh -d CUDA -v

# 启用全部自定义实现
./run.sh -d CUDA -c -m -v
```

## 自定义实现文件

### 核心实现文件

#### 1. Custom NTT
- `src/custom_ntt_hardcoded.cu` - NTT/INTT CUDA 内核实现
- `src/ntt_selector.h` - NTT 动态选择器
- `src/custom_ntt_constants.h` - 共享常量定义

#### 2. Custom MatMul
- `src/custom_matmul_cuda.cu` - 标量域和多项式环矩阵乘法 CUDA 实现
- `src/matmul_selector.h` - MatMul 动态选择器（含类型特征检测）

#### 3. Custom VecOps
- `src/custom_vec_ops_cuda.cu` - 向量操作 CUDA 实现
  - `scalar_mul_vec` - 标量乘向量
  - `vector_add` - 向量加法
  - `vector_mul` - 向量乘法（逐元素相乘）
  - `vector_sum` - 向量求和（两阶段归约）
- `src/vec_ops_selector.h` - VecOps 动态选择器

### 调试辅助文件（stub）
- `icicle/include/custom_ntt_cpu_trace.h` - CPU NTT trace stub（所有函数为空操作）

### 已修改的原有文件
- `src/CMakeLists.txt` - 添加自定义 CUDA 文件到构建系统
- `src/example.cpp` - 添加全局标志和命令行参数解析
- `src/prover.cpp` - 替换为自定义选择器宏
- `src/verifier.cpp` - 替换为自定义选择器宏
- `src/shared.cpp` - 替换为自定义选择器宏
- `src/test_helpers.cpp` - 替换为自定义选择器宏
- `src/types.cpp` - 替换为自定义选择器宏
- `run.sh` - 添加 `-c`, `-m`, `-v` 选项
- `icicle/backend/cpu/include/ntt_task.h` - 使用 stub 版本 trace


## 实现特性

### NTT
- 预计算 twiddle 因子（BabyBear + Koala Bear 双模数）
- Negacyclic NTT（Forward/Inverse）
- 支持批量处理
- 自动回退到 ICICLE 实现（失败时）

### MatMul
- 标量域矩阵乘法（Zq，degree=1）
- 多项式环矩阵乘法（PolyRing，degree=64，逐系数相乘求和）
- 支持转置标志（a_transposed, b_transposed）
- 类型自动检测（C++17 SFINAE）
- 自动回退到 ICICLE 实现

### VecOps
- `scalar_mul_vec`：标量乘向量，支持 stride
- `vector_add`：向量加法
- `vector_mul`：向量乘法（逐元素相乘）
- `vector_sum`：两阶段归约（块内共享内存 + 主机端累加）
- 支持 Zq（degree=1）和 PolyRing（degree=64）
- 自动回退到 ICICLE 实现

### MiscOps
- `decompose/recompose`：平衡分解/重组
- `jl_projection`：Johnson-Lindenstrauss 投影
- `matrix_transpose`：矩阵转置
- 自动回退到 ICICLE 实现

## 性能基准（n=64, r=8, eq=10, cz=10）

| 配置 | Prover 时间 | Verifier 时间 | 总计 | 验证状态 |
|------|-------------|---------------|------|----------|
| ICICLE 默认 | ~132 ms | ~132 ms | ~264 ms | ✓ |
| Custom (NTT) | ~133 ms | ~133 ms | ~266 ms | ✓ |
| Custom (MatMul) | ~230 ms | ~199 ms | ~429 ms | ✓ |
| Custom (VecOps) | ~141 ms | ~141 ms | ~282 ms | ✓ |
| Custom (Misc) | ~173 ms | ~173 ms | ~346 ms | ✓ |
| Custom (all) | ~277 ms | ~277 ms | ~554 ms | ✓ |

**注意**：自定义实现目前性能较慢是因为：
1. 未优化（简单实现验证正确性优先）
2. 频繁的主机-设备内存拷贝
3. 未使用 shared memory tiling
4. 未融合多个小操作


## 优化方向

### 短期优化
1. 减少 H2D/D2H 拷贝，复用 GPU 内存
2. MatMul：使用 shared memory tiling
3. VecOps：向量化访存（float2/uint2）

### 长期优化
1. Kernel 融合（如 scalar_mul + vector_add）
2. 批量处理小矩阵减少 launch overhead
3. 使用 CUDA Streams 并行独立操作
4. 持久化 GPU 数据避免重复传输




