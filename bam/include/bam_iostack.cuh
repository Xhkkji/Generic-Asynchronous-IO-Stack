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
#include <deque>
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
  enum request_state_t : uint32_t
  {
    SUBMITTED = 0,
    READY = 1,
    CONSUMED = 2,
  };

  struct outstanding_meta_t
  {
    uint64_t request_id = 0;
    uint64_t index_ptr = 0;
    int64_t num_index = 0;
    int dim = 0;
    int cache_dim = 0;
    uint64_t key_off = 0;
    s_ctx *ctxs = nullptr;
    uint64_t ctx_count = 0;
    request_state_t state = SUBMITTED;
  };

    // 预先提交几个iter的请求，决定了d_warp_ctxs_array的元素个数
  int presubmit_count = 1;

  // 指向设备端的全局warp上下文数组的数组指针,一个warp持有一个上下文ctx
  // 一个iter持有一个d_warp_ctxs，d_warp_ctxs_array指向d_warp_ctxs的数组
  // s_ctx **d_warp_ctxs_array = nullptr;
  // uint64_t *d_warp_ctxs_array_size = nullptr; // 每个iter的warp上下文数量
  s_ctx *d_warp_ctxs = nullptr;
  std::deque<s_ctx *> d_warp_ctxs_queue;
  std::deque<uint64_t> d_warp_ctxs_count_queue;
  std::deque<outstanding_meta_t> outstanding_queue;
  uint64_t next_request_id = 1;
  uint64_t last_consumed_request_id = 0;

  BaM_IOStack(int presubmit_count = 1) : presubmit_count(presubmit_count) 
  {
    printf("BaM_IOStack constructor: presubmit_count = %d\n", presubmit_count);
    // cudaMalloc(&d_warp_ctxs_array, presubmit_count * sizeof(s_ctx*));
    // cudaMalloc(&d_warp_ctxs_array_size, presubmit_count * sizeof(uint64_t));
  }
  ~BaM_IOStack() = default;

  void push_ctxs(s_ctx *ctxs, uint64_t ctx_count)
  {
    d_warp_ctxs_queue.push_back(ctxs);
    d_warp_ctxs_count_queue.push_back(ctx_count);
    if (d_warp_ctxs == nullptr)
    {
      d_warp_ctxs = ctxs;
    }
  }

  uint64_t register_outstanding(s_ctx *ctxs, uint64_t ctx_count, uint64_t index_ptr,
                                int64_t num_index, int dim, int cache_dim, uint64_t key_off)
  {
    push_ctxs(ctxs, ctx_count);

    outstanding_meta_t meta;
    meta.request_id = next_request_id++;
    meta.index_ptr = index_ptr;
    meta.num_index = num_index;
    meta.dim = dim;
    meta.cache_dim = cache_dim;
    meta.key_off = key_off;
    meta.ctxs = ctxs;
    meta.ctx_count = ctx_count;
    outstanding_queue.push_back(meta);
    return meta.request_id;
  }

  s_ctx *front_ctxs()
  {
    if (d_warp_ctxs_queue.empty())
    {
      d_warp_ctxs = nullptr;
      return nullptr;
    }
    d_warp_ctxs = d_warp_ctxs_queue.front();
    return d_warp_ctxs;
  }

  uint64_t front_ctx_count() const
  {
    if (d_warp_ctxs_count_queue.empty())
    {
      return 0;
    }
    return d_warp_ctxs_count_queue.front();
  }

  const outstanding_meta_t *front_outstanding() const
  {
    if (outstanding_queue.empty())
    {
      return nullptr;
    }
    return &outstanding_queue.front();
  }

  const outstanding_meta_t *outstanding_at(uint64_t offset) const
  {
    if (offset >= outstanding_queue.size())
    {
      return nullptr;
    }
    auto it = outstanding_queue.begin();
    std::advance(it, offset);
    return &(*it);
  }

  s_ctx *ctxs_at(uint64_t offset)
  {
    if (offset >= d_warp_ctxs_queue.size())
    {
      return nullptr;
    }
    auto it = d_warp_ctxs_queue.begin();
    std::advance(it, offset);
    return *it;
  }

  uint64_t outstanding_count() const
  {
    return outstanding_queue.size();
  }

  bool front_outstanding_ready() const
  {
    const outstanding_meta_t *meta = front_outstanding();
    return meta != nullptr && meta->state == READY;
  }

  uint64_t front_ready_request_id() const
  {
    const outstanding_meta_t *meta = front_outstanding();
    if (meta == nullptr || meta->state != READY)
    {
      return 0;
    }
    return meta->request_id;
  }

  request_state_t front_state() const
  {
    const outstanding_meta_t *meta = front_outstanding();
    return meta == nullptr ? CONSUMED : meta->state;
  }

  bool mark_front_ready(uint64_t request_id)
  {
    if (outstanding_queue.empty())
    {
      return false;
    }
    if (request_id != 0 && outstanding_queue.front().request_id != request_id)
    {
      return false;
    }
    outstanding_queue.front().state = READY;
    return true;
  }

  bool mark_ready_at(uint64_t offset, uint64_t request_id)
  {
    if (offset >= outstanding_queue.size())
    {
      return false;
    }
    auto it = outstanding_queue.begin();
    std::advance(it, offset);
    if (request_id != 0 && it->request_id != request_id)
    {
      return false;
    }
    it->state = READY;
    return true;
  }

  bool mark_front_consumed(uint64_t request_id)
  {
    if (outstanding_queue.empty())
    {
      return false;
    }
    if (request_id != 0 && outstanding_queue.front().request_id != request_id)
    {
      return false;
    }
    outstanding_queue.front().state = CONSUMED;
    last_consumed_request_id = outstanding_queue.front().request_id;
    return true;
  }

  uint64_t get_last_consumed_request_id() const
  {
    return last_consumed_request_id;
  }

  void pop_ctxs()
  {
    if (d_warp_ctxs_queue.empty())
    {
      d_warp_ctxs = nullptr;
      return;
    }

    d_warp_ctxs_queue.pop_front();
    d_warp_ctxs_count_queue.pop_front();
    if (!outstanding_queue.empty())
    {
      outstanding_queue.pop_front();
    }
    d_warp_ctxs = d_warp_ctxs_queue.empty() ? nullptr : d_warp_ctxs_queue.front();
  }

  // void read_feature_submit_async(uint64_t tensor_ptr, uint64_t index_ptr,int64_t num_index, int dim, int cache_dim, uint64_t key_off);
  // void read_feature_wait_async(uint64_t tensor_ptr, uint64_t index_ptr,int64_t num_index, int dim, int cache_dim, uint64_t key_off);

};

#endif // 结束
