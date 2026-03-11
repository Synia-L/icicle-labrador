/*****************************************************************************
 * CUDA Memory Pool - 高性能GPU内存管理
 * 
 * 功能：
 * 1. 内存池复用，避免频繁 cudaMalloc/cudaFree
 * 2. 支持异步操作和 CUDA Streams
 * 3. 自动内存对齐
 * 4. 线程安全
 *****************************************************************************/

#ifndef CUDA_MEMORY_POOL_H
#define CUDA_MEMORY_POOL_H

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <iostream>
#include <cstdint>

namespace cuda_opt {

// 内存块信息
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
    cudaStream_t stream;  // 关联的stream
    
    MemoryBlock(void* p, size_t s) 
        : ptr(p), size(s), in_use(false), stream(nullptr) {}
};

// GPU内存池
class CUDAMemoryPool {
private:
    // 按大小分组的内存块池
    std::unordered_map<size_t, std::vector<MemoryBlock*>> size_pools_;
    
    // 所有已分配的内存块（用于清理）
    std::vector<void*> all_allocations_;
    
    // 线程安全锁
    std::mutex pool_mutex_;
    
    // 配置参数
    size_t max_pool_size_;        // 池最大大小（字节）
    size_t current_pool_size_;    // 当前池大小
    bool enable_stats_;           // 是否启用统计
    
    // 统计信息
    size_t total_allocations_;
    size_t cache_hits_;
    size_t cache_misses_;
    
    // 单例实例
    static CUDAMemoryPool* instance_;
    static std::mutex instance_mutex_;
    
    // 私有构造函数
    CUDAMemoryPool(size_t max_pool_size = 1ULL << 30)  // 默认1GB
        : max_pool_size_(max_pool_size)
        , current_pool_size_(0)
        , enable_stats_(true)
        , total_allocations_(0)
        , cache_hits_(0)
        , cache_misses_(0) 
    {}
    
    // 对齐到64字节
    static size_t align_size(size_t size) {
        const size_t alignment = 64;
        return ((size + alignment - 1) / alignment) * alignment;
    }
    
public:
    // 禁止拷贝和赋值
    CUDAMemoryPool(const CUDAMemoryPool&) = delete;
    CUDAMemoryPool& operator=(const CUDAMemoryPool&) = delete;
    
    // 获取单例实例
    static CUDAMemoryPool& get_instance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = new CUDAMemoryPool();
        }
        return *instance_;
    }
    
    // 从池中分配内存
    void* allocate(size_t size, cudaStream_t stream = nullptr) {
        size = align_size(size);
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        total_allocations_++;
        
        // 查找可用的内存块
        auto& pool = size_pools_[size];
        for (auto* block : pool) {
            if (!block->in_use) {
                // 如果有stream关联，等待stream完成
                if (block->stream != nullptr) {
                    cudaStreamSynchronize(block->stream);
                }
                
                block->in_use = true;
                block->stream = stream;
                cache_hits_++;
                
                if (enable_stats_ && total_allocations_ % 1000 == 0) {
                    print_stats();
                }
                
                return block->ptr;
            }
        }
        
        // 没有可用块，分配新内存
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        
        if (err != cudaSuccess) {
            std::cerr << "[MEMORY POOL] cudaMalloc failed: " 
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        
        cache_misses_++;
        current_pool_size_ += size;
        all_allocations_.push_back(ptr);
        
        auto* block = new MemoryBlock(ptr, size);
        block->in_use = true;
        block->stream = stream;
        pool.push_back(block);
        
        return ptr;
    }
    
    // 归还内存到池
    void deallocate(void* ptr, size_t size, cudaStream_t stream = nullptr) {
        if (!ptr) return;
        
        size = align_size(size);
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        auto& pool = size_pools_[size];
        for (auto* block : pool) {
            if (block->ptr == ptr) {
                block->in_use = false;
                block->stream = stream;
                return;
            }
        }
        
        // 如果找不到，说明不是从池分配的，直接释放
        cudaFree(ptr);
    }
    
    // 清空池（释放所有未使用的内存）
    void clear_unused() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        for (auto& [size, pool] : size_pools_) {
            auto it = pool.begin();
            while (it != pool.end()) {
                if (!(*it)->in_use) {
                    cudaFree((*it)->ptr);
                    current_pool_size_ -= (*it)->size;
                    delete *it;
                    it = pool.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    
    // 销毁池（释放所有内存）
    void destroy() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        for (auto* ptr : all_allocations_) {
            cudaFree(ptr);
        }
        
        for (auto& [size, pool] : size_pools_) {
            for (auto* block : pool) {
                delete block;
            }
        }
        
        size_pools_.clear();
        all_allocations_.clear();
        current_pool_size_ = 0;
    }
    
    // 打印统计信息
    void print_stats() const {
        if (!enable_stats_) return;
        
        float hit_rate = total_allocations_ > 0 
            ? (100.0f * cache_hits_ / total_allocations_) 
            : 0.0f;
            
        std::cout << "[MEMORY POOL] Stats: "
                  << "Allocations=" << total_allocations_ << ", "
                  << "Hits=" << cache_hits_ << ", "
                  << "Misses=" << cache_misses_ << ", "
                  << "Hit Rate=" << hit_rate << "%, "
                  << "Pool Size=" << (current_pool_size_ >> 20) << " MB"
                  << std::endl;
    }
    
    // 析构函数
    ~CUDAMemoryPool() {
        destroy();
    }
};

// RAII 风格的内存管理
class CUDAMemoryGuard {
private:
    void* ptr_;
    size_t size_;
    cudaStream_t stream_;
    CUDAMemoryPool& pool_;
    
public:
    CUDAMemoryGuard(size_t size, cudaStream_t stream = nullptr)
        : size_(size)
        , stream_(stream)
        , pool_(CUDAMemoryPool::get_instance())
    {
        ptr_ = pool_.allocate(size, stream);
    }
    
    ~CUDAMemoryGuard() {
        pool_.deallocate(ptr_, size_, stream_);
    }
    
    void* get() const { return ptr_; }
    operator void*() const { return ptr_; }
    
    // 禁止拷贝
    CUDAMemoryGuard(const CUDAMemoryGuard&) = delete;
    CUDAMemoryGuard& operator=(const CUDAMemoryGuard&) = delete;
    
    // 支持移动
    CUDAMemoryGuard(CUDAMemoryGuard&& other) noexcept
        : ptr_(other.ptr_)
        , size_(other.size_)
        , stream_(other.stream_)
        , pool_(other.pool_)
    {
        other.ptr_ = nullptr;
    }
};

// 静态成员初始化
inline CUDAMemoryPool* CUDAMemoryPool::instance_ = nullptr;
inline std::mutex CUDAMemoryPool::instance_mutex_;

} // namespace cuda_opt

#endif // CUDA_MEMORY_POOL_H

