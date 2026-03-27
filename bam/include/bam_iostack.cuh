#ifndef BAM_IOSTACK_H  // 如果 BAM_IOSTACK_H 没有定义过
#define BAM_IOSTACK_H  // 那就定义它，并包含下面的内容

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include "util.h"
#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "buffer.h"
#include "ctrl.h"
#include <iostream>
#include "nvm_parallel_queue.h"
#include "nvm_cmd.h"

#include "window_buffer.h"

// 新增
#include <atomic>  // std::atomic（如果支持）
#include <cstdint> // 标准整数类型
// CUDA相关
#include <cuda_runtime.h> // CUDA运行时
// #include <cuda_atomic.h>   // CUDA原子操作（如果使用）

// 使用共享内存传递上下文

struct s_ctx
{
    int idx_idx;  // warp在核函数中的全局索引
    uint64_t row_index;  // 使得wait和submit统一，即令i和idx_idx统一

    // 保存bam指针的参数
    void *bam_page = nullptr;
    void *array = nullptr;
    size_t start = 0;
    size_t end = 0;
    int64_t range_id = -1;
    void* addr;
    
    // page, start, end, range_id是全局变量
    uint32_t eq_mask;
    int master;
    uint32_t count;
    uint64_t base_master;
    uint32_t ctrl; // 控制器ID（submit选择的）
    uint32_t queue;

    // 修改：自加， 用于异步
    uint16_t cid;
    nvm_cmd_t cmd;
    uint32_t page_trans;

    // 成员函数acquire_page
    uint64_t member_acquire_ctrl;
    uint64_t member_acquire_b_page;
    Controller *member_acquire_c;
    QueuePair *member_acquire_cAddq;

    // 全局函数acquire_page
    int64_t r;
    uint64_t page;  // 需要，用于给bam指针的成员变量page赋值，用于引用计数的释放
    uint64_t subindex;
    uint64_t gaddr;
    uint64_t observed_page_translation;

    bool isHit;// 若该ctx对应的warp中32个线程全部命中，则为true，否则为false
    // bool islaneHit[32] = {false};// 每个线程的命中情况

    // 添加构造函数确保初始化
    __device__ s_ctx() : addr(0), r(-1), eq_mask(0), master(-1),
                         count(0), base_master(0), observed_page_translation(0), isHit(false) {}
};

template <typename T>
struct BaM_IOStack 
{
    // 预先提交几个iter的请求，决定了d_warp_ctxs_array的元素个数
  int presubmit_count = 1;

  // 指向设备端的全局warp上下文数组的数组指针,一个warp持有一个上下文ctx
  // 一个iter持有一个d_warp_ctxs，d_warp_ctxs_array指向d_warp_ctxs的数组
  // s_ctx **d_warp_ctxs_array = nullptr;
  // uint64_t *d_warp_ctxs_array_size = nullptr; // 每个iter的warp上下文数量
  s_ctx *d_warp_ctxs = nullptr;

  BaM_IOStack(int presubmit_count = 1) : presubmit_count(presubmit_count) 
  {
    printf("BaM_IOStack constructor: presubmit_count = %d\n", presubmit_count);
    // cudaMalloc(&d_warp_ctxs_array, presubmit_count * sizeof(s_ctx*));
    // cudaMalloc(&d_warp_ctxs_array_size, presubmit_count * sizeof(uint64_t));
  }
  ~BaM_IOStack() = default;

  // void read_feature_submit_async(uint64_t tensor_ptr, uint64_t index_ptr,int64_t num_index, int dim, int cache_dim, uint64_t key_off);
  // void read_feature_wait_async(uint64_t tensor_ptr, uint64_t index_ptr,int64_t num_index, int dim, int cache_dim, uint64_t key_off);

};

#endif // 结束