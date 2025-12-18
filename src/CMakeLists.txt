cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# 启用CUDA支持
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

project(example LANGUAGES CXX CUDA)

# Labrador 主程序
add_executable(example
  example.cpp
  utils.cpp 
  prover.cpp
  verifier.cpp
  shared.cpp
  types.cpp
  oracle.cpp
  test_helpers.cpp
  benchmarking.cpp
  custom_ntt_hardcoded.cu
  custom_matmul_cuda.cu
  custom_vec_ops_cuda.cu
  custom_misc_ops_cuda.cu
)

install(TARGETS example
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")

# Include 路径
target_include_directories(example PRIVATE 
    "../icicle/include"
    "../icicle/backend/cuda/include"
    "../icicle/backend/cpu/include"
)

# Link 路径和库
target_link_directories(example PRIVATE "${CMAKE_SOURCE_DIR}/../build/icicle")
target_link_libraries(example PRIVATE icicle_ring_babykoala icicle_device icicle_hash cudart)

target_link_options(example PRIVATE -rdynamic)

# CUDA 属性
set_target_properties(example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89"
)

# 编译定义
target_compile_definitions(example PRIVATE RING)

if(BACKEND_DIR)
  add_compile_definitions(BACKEND_DIR="${BACKEND_DIR}")
endif()
