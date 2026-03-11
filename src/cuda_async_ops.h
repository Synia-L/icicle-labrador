/*****************************************************************************
 * CUDA Async Operations - 异步操作和批量处理框架
 * 
 * 功能：
 * 1. 支持异步内存传输
 * 2. 支持 CUDA Streams 并行
 * 3. 批量操作接口
 * 4. 持久化GPU数据选项
 *****************************************************************************/

#ifndef CUDA_ASYNC_OPS_H
#define CUDA_ASYNC_OPS_H

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <functional>
#include "cuda_memory_pool.h"

namespace cuda_opt {

// Stream池管理
class CUDAStreamPool {
private:
    std::vector<cudaStream_t> streams_;
    size_t next_stream_idx_;
    std::mutex stream_mutex_;
    
    static CUDAStreamPool* instance_;
    static std::mutex instance_mutex_;
    
    CUDAStreamPool(size_t num_streams = 4) : next_stream_idx_(0) {
        streams_.resize(num_streams);
        for (size_t i = 0; i < num_streams; ++i) {
            cudaStreamCreate(&streams_[i]);
        }
    }
    
public:
    static CUDAStreamPool& get_instance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = new CUDAStreamPool();
        }
        return *instance_;
    }
    
    cudaStream_t get_stream() {
        std::lock_guard<std::mutex> lock(stream_mutex_);
        cudaStream_t stream = streams_[next_stream_idx_];
        next_stream_idx_ = (next_stream_idx_ + 1) % streams_.size();
        return stream;
    }
    
    void synchronize_all() {
        for (auto stream : streams_) {
            cudaStreamSynchronize(stream);
        }
    }
    
    ~CUDAStreamPool() {
        for (auto stream : streams_) {
            cudaStreamDestroy(stream);
        }
    }
};

inline CUDAStreamPool* CUDAStreamPool::instance_ = nullptr;
inline std::mutex CUDAStreamPool::instance_mutex_;

// 持久化GPU缓冲区（数据保持在GPU上）
template<typename T>
class PersistentGPUBuffer {
private:
    void* d_ptr_;
    size_t size_;
    size_t capacity_;
    cudaStream_t stream_;
    bool owns_memory_;
    
public:
    PersistentGPUBuffer(size_t initial_capacity = 0)
        : d_ptr_(nullptr)
        , size_(0)
        , capacity_(0)
        , stream_(nullptr)
        , owns_memory_(true)
    {
        if (initial_capacity > 0) {
            resize(initial_capacity);
        }
    }
    
    ~PersistentGPUBuffer() {
        if (owns_memory_ && d_ptr_) {
            auto& pool = CUDAMemoryPool::get_instance();
            pool.deallocate(d_ptr_, capacity_ * sizeof(T), stream_);
        }
    }
    
    void resize(size_t new_size) {
        if (new_size <= capacity_) {
            size_ = new_size;
            return;
        }
        
        // 需要重新分配
        auto& pool = CUDAMemoryPool::get_instance();
        void* new_ptr = pool.allocate(new_size * sizeof(T), stream_);
        
        if (!new_ptr) {
            throw std::runtime_error("Failed to allocate GPU memory");
        }
        
        // 拷贝旧数据
        if (d_ptr_ && size_ > 0) {
            cudaMemcpyAsync(new_ptr, d_ptr_, size_ * sizeof(T), 
                           cudaMemcpyDeviceToDevice, stream_);
        }
        
        // 释放旧内存
        if (d_ptr_) {
            pool.deallocate(d_ptr_, capacity_ * sizeof(T), stream_);
        }
        
        d_ptr_ = new_ptr;
        capacity_ = new_size;
        size_ = new_size;
    }
    
    // 异步上传数据
    void upload_async(const T* host_data, size_t count, cudaStream_t stream = nullptr) {
        if (count > capacity_) {
            resize(count);
        }
        
        stream_ = stream ? stream : CUDAStreamPool::get_instance().get_stream();
        cudaMemcpyAsync(d_ptr_, host_data, count * sizeof(T), 
                       cudaMemcpyHostToDevice, stream_);
        size_ = count;
    }
    
    // 异步下载数据
    void download_async(T* host_data, cudaStream_t stream = nullptr) const {
        cudaStream_t s = stream ? stream : stream_;
        cudaMemcpyAsync(host_data, d_ptr_, size_ * sizeof(T), 
                       cudaMemcpyDeviceToHost, s);
    }
    
    void synchronize() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
        }
    }
    
    void* get() const { return d_ptr_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    cudaStream_t get_stream() const { return stream_; }
    
    // 禁止拷贝
    PersistentGPUBuffer(const PersistentGPUBuffer&) = delete;
    PersistentGPUBuffer& operator=(const PersistentGPUBuffer&) = delete;
    
    // 支持移动
    PersistentGPUBuffer(PersistentGPUBuffer&& other) noexcept
        : d_ptr_(other.d_ptr_)
        , size_(other.size_)
        , capacity_(other.capacity_)
        , stream_(other.stream_)
        , owns_memory_(other.owns_memory_)
    {
        other.d_ptr_ = nullptr;
        other.owns_memory_ = false;
    }
};

// 批量操作类型
enum class BatchOpType {
    SCALAR_MUL,
    VECTOR_ADD,
    VECTOR_MUL,
    VECTOR_SUB,
    MATRIX_MUL,
    TRANSPOSE
};

// 批量操作描述
struct BatchOperation {
    BatchOpType type;
    void* inputs[4];      // 最多4个输入
    void* output;
    size_t sizes[4];      // 相关尺寸
    uint32_t params[4];   // 参数（如degree, stride等）
    
    BatchOperation() {
        for (int i = 0; i < 4; ++i) {
            inputs[i] = nullptr;
            sizes[i] = 0;
            params[i] = 0;
        }
        output = nullptr;
    }
};

// 批量操作执行器
class BatchExecutor {
private:
    std::vector<BatchOperation> pending_ops_;
    cudaStream_t stream_;
    std::mutex ops_mutex_;
    
public:
    BatchExecutor(cudaStream_t stream = nullptr) 
        : stream_(stream ? stream : CUDAStreamPool::get_instance().get_stream()) 
    {}
    
    void add_operation(const BatchOperation& op) {
        std::lock_guard<std::mutex> lock(ops_mutex_);
        pending_ops_.push_back(op);
    }
    
    // 执行所有待处理操作
    int execute_all();
    
    void clear() {
        std::lock_guard<std::mutex> lock(ops_mutex_);
        pending_ops_.clear();
    }
    
    size_t pending_count() const {
        return pending_ops_.size();
    }
    
    cudaStream_t get_stream() const { return stream_; }
};

// 异步操作上下文
struct AsyncContext {
    cudaStream_t stream;
    void* d_input;
    void* d_output;
    void* d_temp;
    size_t input_size;
    size_t output_size;
    size_t temp_size;
    bool owns_device_memory;
    
    AsyncContext() 
        : stream(nullptr)
        , d_input(nullptr)
        , d_output(nullptr)
        , d_temp(nullptr)
        , input_size(0)
        , output_size(0)
        , temp_size(0)
        , owns_device_memory(false)
    {}
    
    ~AsyncContext() {
        if (owns_device_memory) {
            auto& pool = CUDAMemoryPool::get_instance();
            if (d_input) pool.deallocate(d_input, input_size, stream);
            if (d_output) pool.deallocate(d_output, output_size, stream);
            if (d_temp) pool.deallocate(d_temp, temp_size, stream);
        }
    }
    
    void allocate_buffers(size_t in_size, size_t out_size, size_t tmp_size = 0) {
        auto& pool = CUDAMemoryPool::get_instance();
        stream = CUDAStreamPool::get_instance().get_stream();
        
        input_size = in_size;
        output_size = out_size;
        temp_size = tmp_size;
        
        d_input = pool.allocate(input_size, stream);
        d_output = pool.allocate(output_size, stream);
        if (temp_size > 0) {
            d_temp = pool.allocate(temp_size, stream);
        }
        
        owns_device_memory = true;
    }
    
    void upload_input_async(const void* host_data) {
        cudaMemcpyAsync(d_input, host_data, input_size, 
                       cudaMemcpyHostToDevice, stream);
    }
    
    void download_output_async(void* host_data) {
        cudaMemcpyAsync(host_data, d_output, output_size, 
                       cudaMemcpyDeviceToHost, stream);
    }
    
    void synchronize() {
        if (stream) {
            cudaStreamSynchronize(stream);
        }
    }
};

// 便捷函数：创建异步上下文
inline std::unique_ptr<AsyncContext> create_async_context(
    size_t input_size, 
    size_t output_size, 
    size_t temp_size = 0
) {
    auto ctx = std::make_unique<AsyncContext>();
    ctx->allocate_buffers(input_size, output_size, temp_size);
    return ctx;
}

} // namespace cuda_opt

#endif // CUDA_ASYNC_OPS_H

