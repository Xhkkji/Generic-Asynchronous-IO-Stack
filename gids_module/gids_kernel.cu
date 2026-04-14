

template <typename T = float>
__global__ void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off)
{

  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx)
  {
    bam_ptr<T> ptr(dr);

    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;

    for (; tid < dim; tid += 32)
    {
      // T temp = ptr[(row_index) * cache_dim + tid];
      const size_t idx = (row_index)*cache_dim + tid;
      // out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = ptr[idx];
      out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = ptr.read(idx);
      
      // printf("read_feature_kernel idx:%llu\n", (unsigned long long)idx);
      // T temp = ptr.read_submit_async((row_index)*cache_dim + tid); // √
      // if (!ptr.ctx.isHit)
      // {
      //   temp = ptr.read_wait_async((row_index)*cache_dim + tid); // √
      //   ptr.ctx.isHit = true;                                    // 重要
      // }
      // out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    }
  }
}

template <typename T = float>
__global__ void read_feature_kernel_submit_async(array_d_t<T> *dr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_warp_ctxs)
{
  uint64_t bid = blockIdx.x;  // 获取当前block的索引（0到g_size-1）
  int num_warps = blockDim.x / 32;  // 计算每个block中的warp数量（128/32=4）
  int warp_id = threadIdx.x / 32;  // 由于一个block有128个线程拆成4个warp，计算当前线程所在的warp在block内的ID（0-3）
  int idx_idx = bid * num_warps + warp_id;  // 当前warp处理的全局特征索引
  int lane_id = threadIdx.x % 32;
  // if(lane_id == 0)
  // {
  //   // printf("warp idx_idx:%d\n", idx_idx);
  //   // printf("submit_async launched, idx_idx=%d\n", idx_idx);
  // }
  // if(idx_idx == 40000)
  // {
  //   printf("d_warp_ctxs:%p\n", d_warp_ctxs);
  // }
  
  if (idx_idx < num_idx)
  {
    bam_ptr<T> ptr(dr, false);

    uint64_t row_index = index_ptr[idx_idx] + key_off;
    // if(lane_id == 0 && idx_idx < 30)
    // {
    //   printf("submit_async idx_idx:%d, row_index:%llu\n", idx_idx, (unsigned long long)row_index);
    // }
    uint64_t tid = threadIdx.x % 32;
    // printf("submit idx_idx:%d\n", idx_idx);
    // s_ctx& ctx = d_warp_ctxs[idx_idx];
    // if(lane_id == 0)
    // {
    //   ctx.idx_idx = idx_idx;  // 重要，确保submit和wait使用同一个idx_idx
    //   ctx.row_index = row_index;  // 重要，确保submit和wait使用同一个row_index
    //   // printf("submit_async idx_idx:%d, row_index:%llu\n", ctx.idx_idx, (unsigned long long)ctx.row_index);
    // }
    // __threadfence();  // ⭐ 内存屏障：强制刷新所有内存操作到全局内存，确保其他线程能看到最新的ctx状态
    __syncwarp();
    
    // printf("warp %d 获取ctx地址:%p\n", idx_idx, &ctx);
    // printf("获取ctx完成..\n");
    // 每个线程内，分别取出自己对应负责的特征
    int loop_idx = 0;
    for (; tid < dim; tid += 32)
    {
      s_ctx& ctx = d_warp_ctxs[idx_idx * 32 + loop_idx++];
      // T temp = ptr[(row_index) * cache_dim + tid];
      const size_t idx = (row_index)*cache_dim + tid;
      // out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = ptr[idx];
      // out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = ptr.read(idx);

      ptr.read_submit_async((row_index)*cache_dim + tid, ctx); // √
      // 是否命中是以warp为单位的，isHIt只在warp的主线程中赋值
      __syncwarp();
      // 一切数据由wait函数获取，submit函数不进行数据写回，避免数据不一致问题

    }
    // __syncthreads();  // 自加
  }
}

template <typename T = float>
__global__ void read_feature_kernel_wait_async(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_warp_ctxs)
{
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;

  // if(lane_id == 0)
  // {
  //   // printf("d_warp:%p\n", d_warp_ctxs);
  //   printf("idx_idx:%d\n", idx_idx);
  //   // printf("read_feature_kernel_wait_async launched..\n");
  // }
  // if(idx_idx == 0)
  // {
  //   printf("d_warp_ctxs:%p\n", d_warp_ctxs);
  // }

  int loop_idx = 0;
  if (idx_idx < num_idx)
  {
    bam_ptr<T> ptr(dr, true);

    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;

    // Materialize all rows in the wait stage. Miss rows complete the pending IO,
    // while hit rows rebuild the cache pointer from the submit-time context and
    // overwrite any speculative submit-stage values.
    for (; tid < dim; tid += 32)
    {
      // if(tid == 0)
      // printf("wait (row_index)*cache_dim + tid:%d\n", (row_index)*cache_dim + tid);
      s_ctx& ctx = d_warp_ctxs[idx_idx * 32 + loop_idx++];
      T temp = ptr.read_wait_async((row_index)*cache_dim + tid, ctx);
      out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    }
  }
}

template <typename T = float>
__global__ void read_feature_kernel_single_thread_poll(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_warp_ctxs)
{
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  int lane_id = threadIdx.x % 32;

  // 以 warp 为粒度处理一个结点，warp 内每个线程负责一个 32 维 chunk 的轮询。
  if (idx_idx < num_idx)
  {
    bam_ptr<T> ptr(dr, false);
    uint64_t row_index = index_ptr[idx_idx] + key_off;
    int num_chunks = (dim + 31) / 32;
    int chunk_idx = lane_id;

    for (; chunk_idx < num_chunks; chunk_idx += 32)
    {
      const uint64_t submit_idx = (row_index)*cache_dim + static_cast<uint64_t>(chunk_idx) * 32;
      s_ctx& ctx = d_warp_ctxs[idx_idx * 32 + chunk_idx];
      int64_t range_idx = dr->find_range(submit_idx);
      if (range_idx == -1)
      {
        continue;
      }

      auto range = dr->d_ranges + range_idx;
      const uint64_t page = range->get_page(submit_idx);
      bool is_page_owner = (chunk_idx == 0);
      if (!is_page_owner)
      {
        const uint64_t prev_submit_idx = submit_idx - 32;
        const int64_t prev_range_idx = dr->find_range(prev_submit_idx);
        if (prev_range_idx == -1)
        {
          is_page_owner = true;
        }
        else
        {
          auto prev_range = dr->d_ranges + prev_range_idx;
          const uint64_t prev_page = prev_range->get_page(prev_submit_idx);
          is_page_owner = (prev_range_idx != range_idx) || (prev_page != page);
        }
      }

      if (is_page_owner)
      {
        ptr.read_single_thread_poll(submit_idx, ctx);
      }
    }
  }
}
// 假设一个结点的所有特征都在同一页上
template <typename T = float>
__global__ void read_feature_kernel_single_page_single_thread_poll(array_d_t<T> *dr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_warp_ctxs)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 单页假设版：一个线程处理一个结点的一次轮询，不再占用同一个 warp 的其他线程。
  if (global_tid < num_idx)
  {
    bam_ptr<T> ptr(dr, false);
    uint64_t row_index = index_ptr[global_tid] + key_off;
    const uint64_t submit_idx = (row_index) * cache_dim;
    s_ctx& ctx = d_warp_ctxs[global_tid * 32];
    ptr.read_single_thread_poll(submit_idx, ctx);
  }
}

template <typename T = float>
__global__ void read_feature_kernel_single_page_single_thread_poll_rowctx(array_d_t<T> *dr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_row_ctxs)
{
  (void)dim;
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_tid < static_cast<uint32_t>(num_idx))
  {
    bam_ptr<T> ptr(dr, false);
    uint64_t row_index = index_ptr[global_tid] + key_off;
    const uint64_t submit_idx = (row_index) * cache_dim;
    s_ctx& ctx = d_row_ctxs[global_tid];
    ptr.read_single_thread_poll(submit_idx, ctx);
  }
}

template <typename T = float>
__global__ void read_feature_kernel_single_page_single_thread_try_poll(array_d_t<T> *dr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_warp_ctxs,
                                    uint32_t *pending_count)
{
  (void)dim;
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_tid < num_idx)
  {
    bam_ptr<T> ptr(dr, false);
    uint64_t row_index = index_ptr[global_tid] + key_off;
    const uint64_t submit_idx = (row_index) * cache_dim;
    s_ctx& ctx = d_warp_ctxs[global_tid * 32];
    if (!ptr.read_single_thread_try_poll(submit_idx, ctx))
    {
      atomicAdd(pending_count, 1U);
    }
  }
}

template <typename T = float>
__global__ void read_feature_kernel_single_page_single_thread_try_poll_serial(array_d_t<T> *dr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_warp_ctxs,
                                    uint32_t *pending_count)
{
  (void)dim;
  if (blockIdx.x != 0 || threadIdx.x != 0)
  {
    return;
  }

  bam_ptr<T> ptr(dr, false);
  uint32_t local_pending = 0;
  for (int64_t row = 0; row < num_idx; ++row)
  {
    uint64_t row_index = index_ptr[row] + key_off;
    const uint64_t submit_idx = row_index * cache_dim;
    s_ctx &ctx = d_warp_ctxs[row * 32];
    if (!ptr.read_single_thread_try_poll(submit_idx, ctx))
    {
      ++local_pending;
    }
  }
  *pending_count = local_pending;
}

template <typename T = float>
__global__ void read_feature_kernel_submit_async_rowctx(array_d_t<T> *dr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_row_ctxs)
{
  (void)dim;
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int idx_idx = bid * num_warps + warp_id;

  if (idx_idx >= num_idx || lane_id != 0)
  {
    return;
  }

  bam_ptr<T> ptr(dr, false);
  uint64_t row_index = index_ptr[idx_idx] + key_off;
  const uint64_t submit_idx = row_index * cache_dim;
  s_ctx &ctx = d_row_ctxs[idx_idx];
  ptr.read_submit_async(submit_idx, ctx);
}

template <typename T = float>
__global__ void read_feature_kernel_get_feature_light_rowctx(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_row_ctxs)
{
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= static_cast<uint32_t>(num_idx))
  {
    return;
  }

  bam_ptr<T> ptr(dr, true);
  uint64_t row_index = index_ptr[row] + key_off;
  s_ctx &ctx = d_row_ctxs[row];
  for (int tid = 0; tid < dim; ++tid)
  {
    out_tensor_ptr[static_cast<uint64_t>(row) * dim + tid] =
        ptr.read_post_poll_light(static_cast<uint64_t>(row_index) * cache_dim + tid, ctx);
  }
}

template <typename T = float>
__device__ bool finalize_registered_ctx_completion(array_d_t<T> *dr, s_ctx &ctx)
{
  if (ctx.isHit || ctx.r < 0)
  {
    return false;
  }

  auto range = dr->d_ranges + ctx.r;
  data_page_t &page = range->pages[ctx.page];
  const uint64_t read_state = page.state.load(simt::memory_order_acquire);
  const uint64_t st = (read_state >> (CNT_SHIFT + 1)) & 0x03;

  if (st == V_NB)
  {
    const uint32_t page_trans = page.offset;
    ctx.page_trans = page_trans;
    ctx.observed_page_translation = (page_trans == ASYNC_PAGE_TRANS_PENDING || page_trans >= range->cache.n_pages)
                                        ? 0
                                        : range->cache.cache_pages[page_trans].page_translation;
    ctx.isHit = true;
    ctx.has_pending_io = 0;
    page.polling_flag.store(0, simt::memory_order_release);
    return true;
  }

  if (st != NV_B)
  {
    return false;
  }

  page.offset = ctx.page_trans;
  ctx.observed_page_translation = (ctx.page_trans == ASYNC_PAGE_TRANS_PENDING || ctx.page_trans >= range->cache.n_pages)
                                      ? 0
                                      : range->cache.cache_pages[ctx.page_trans].page_translation;
  range->miss_cnt.fetch_add(ctx.count, simt::memory_order_relaxed);
  page.state.fetch_xor(DISABLE_BUSY_ENABLE_VALID, simt::memory_order_release);
  page.polling_flag.store(0, simt::memory_order_release);
  ctx.isHit = true;
  ctx.has_pending_io = 0;
  return true;
}

template <typename T = float>
__device__ bool refresh_registered_ctx_from_valid_page(array_d_t<T> *dr, s_ctx &ctx)
{
  if (ctx.isHit || ctx.r < 0)
  {
    return ctx.isHit;
  }

  auto range = dr->d_ranges + ctx.r;
  data_page_t &page = range->pages[ctx.page];
  const uint64_t read_state = page.state.load(simt::memory_order_acquire);
  const uint64_t st = (read_state >> (CNT_SHIFT + 1)) & 0x03;
  if (st != V_NB)
  {
    return false;
  }

  const uint32_t page_trans = page.offset;
  if (page_trans == ASYNC_PAGE_TRANS_PENDING || page_trans >= range->cache.n_pages)
  {
    return false;
  }

  ctx.page_trans = page_trans;
  ctx.observed_page_translation = range->cache.cache_pages[page_trans].page_translation;
  ctx.isHit = true;
  page.polling_flag.store(0, simt::memory_order_release);
  return true;
}

template <typename T = float>
__global__ void register_registered_ctx_lookup_kernel(array_d_t<T> *dr,
                                                      s_ctx *d_row_ctxs,
                                                      int64_t num_index,
                                                      uint32_t n_ctrls,
                                                      uint32_t cid_capacity,
                                                      s_ctx **ctx_lookup)
{
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= static_cast<uint32_t>(num_index))
  {
    return;
  }

  s_ctx &ctx = d_row_ctxs[row];
  if (ctx.r < 0 || ctx.isHit || ctx.has_pending_io == 0)
  {
    return;
  }

  uint32_t logical_queue_idx = ctx.queue;
  for (uint32_t ctrl_idx = 0; ctrl_idx < ctx.ctrl && ctrl_idx < n_ctrls; ++ctrl_idx)
  {
    Controller *candidate = dr->d_ranges[0].cache.d_ctrls[ctrl_idx];
    if (candidate == nullptr)
    {
      return;
    }
    logical_queue_idx += candidate->n_qps;
  }

  if (ctx.cid >= cid_capacity)
  {
    return;
  }

  const uint64_t slot = static_cast<uint64_t>(logical_queue_idx) * cid_capacity + ctx.cid;
  ctx_lookup[slot] = &ctx;
}

template <typename T = float>
__global__ void service_registered_cq_window_kernel(array_d_t<T> *dr,
                                                    uint32_t n_ctrls,
                                                    uint32_t total_logical_queues,
                                                    uint32_t *event_count,
                                                    uint32_t *progress_count,
                                                    uint32_t *lookup_miss_count,
                                                    uint32_t max_events_per_queue,
                                                    uint32_t cid_capacity,
                                                    s_ctx **ctx_lookup)
{
  if (ctx_lookup == nullptr)
  {
    return;
  }

  uint32_t logical_queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (logical_queue_idx >= total_logical_queues)
  {
    return;
  }

  Controller *ctrl = nullptr;
  uint32_t local_queue_idx = logical_queue_idx;
  uint32_t ctrl_idx = 0;
  for (; ctrl_idx < n_ctrls; ++ctrl_idx)
  {
    Controller *candidate = dr->d_ranges[0].cache.d_ctrls[ctrl_idx];
    if (candidate == nullptr)
    {
      return;
    }
    if (local_queue_idx < candidate->n_qps)
    {
      ctrl = candidate;
      break;
    }
    local_queue_idx -= candidate->n_qps;
  }

  if (ctrl == nullptr || local_queue_idx >= ctrl->n_qps)
  {
    return;
  }

  QueuePair *qp = (ctrl->d_qps) + local_queue_idx;
  uint32_t handled = 0;
  while (handled < max_events_per_queue)
  {
    uint16_t cid = 0;
    uint32_t cq_pos = 0;
    uint32_t loc = 0;
    uint32_t head = 0;
    if (!cq_try_peek_head(&qp->cq, &cid, &cq_pos, &loc, &head))
    {
      break;
    }

    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, loc, head);
    put_cid(&qp->sq, cid);
    atomicAdd(event_count, 1U);
    ++handled;

    if (cid >= cid_capacity)
    {
      atomicAdd(lookup_miss_count, 1U);
      continue;
    }

    const uint64_t slot = static_cast<uint64_t>(logical_queue_idx) * cid_capacity + cid;
    s_ctx *ctx_ptr = ctx_lookup[slot];
    if (ctx_ptr == nullptr)
    {
      atomicAdd(lookup_miss_count, 1U);
      continue;
    }

    if (finalize_registered_ctx_completion(dr, *ctx_ptr))
    {
      atomicAdd(progress_count, 1U);
    }
    ctx_lookup[slot] = nullptr;
  }
}

template <typename T = float>
__global__ void check_registered_request_ready_kernel(array_d_t<T> *dr,
                                                      s_ctx *d_warp_ctxs,
                                                      int64_t num_index,
                                                      uint32_t ctx_stride,
                                                      uint32_t *pending_count)
{
  uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_tid >= static_cast<uint32_t>(num_index))
  {
    return;
  }

  s_ctx &ctx = d_warp_ctxs[static_cast<uint64_t>(global_tid) * ctx_stride];
  if (!ctx.isHit)
  {
    refresh_registered_ctx_from_valid_page(dr, ctx);
  }
  if (!ctx.isHit)
  {
    atomicAdd(pending_count, 1U);
  }
}
// 单独获取，走状态机
template <typename T = float>
__global__ void read_feature_kernel_get_feature(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_warp_ctxs)
{
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;

  int loop_idx = 0;

  if (idx_idx < num_idx)
  {
    bam_ptr<T> ptr(dr, true);

    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;

    // Materialize all rows in the wait stage. Miss rows complete the pending IO,
    // while hit rows rebuild the cache pointer from the submit-time context and
    // overwrite any speculative submit-stage values.
    for (; tid < dim; tid += 32)
    {
      s_ctx& ctx = d_warp_ctxs[idx_idx * 32 + loop_idx++];
      T temp = ptr.read_single_thread_get((row_index)*cache_dim + tid, ctx);
      out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    }
  }
}
// 轻量化单独获取，直接从cache槽位返回数据，不走状态机，不区分命中和未命中，适用于已经轮询完成的行
template <typename T = float>
__global__ void read_feature_kernel_get_feature_light(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, s_ctx* d_warp_ctxs)
{
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;

  int loop_idx = 0;

  if (idx_idx < num_idx)
  {
    bam_ptr<T> ptr(dr, true);

    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;

    for (; tid < dim; tid += 32)
    {
      s_ctx& ctx = d_warp_ctxs[idx_idx * 32 + loop_idx++];
      T temp = ptr.read_post_poll_light((row_index)*cache_dim + tid, ctx);
      out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    }
  }
}



template <typename T = float>
__global__ void read_feature_kernel_with_cpu_backing_memory(array_d_t<T> *dr, range_d_t<T> *range, T *out_tensor_ptr,
                                                            int64_t *index_ptr, int dim,
                                                            int64_t num_idx, int cache_dim, GIDS_CPU_buffer<T> CPU_buffer, bool cpu_seq, unsigned int *d_cpu_access, uint64_t key_off)
{
  // out_tensor_ptr:[index_num[i], feature_dim]二维数组指针
  // printf("read_feature_kernel_with_cpu_backing_memory\n");
  uint64_t bid = blockIdx.x;

  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;

  // printf("warp_id:%d\n", warp_id);
  // printf("Thread:%d\n", blockIdx.x * blockDim.x + threadIdx.x);
  if (idx_idx < num_idx)
  {
    bam_ptr<T> ptr(dr);

    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;

    uint32_t cpu_off = range->get_cpu_offset(row_index);
    if (cpu_seq)
    {
      // CPU内存路径
      if (row_index < CPU_buffer.cpu_buffer_len)
      {
        if (tid == 0)
          atomicAdd(d_cpu_access, 1);
        for (; tid < dim; tid += 32)
        {
          T temp = CPU_buffer.device_cpu_buffer[(row_index)*cache_dim + tid];
          out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
      }
      // SSD内存路径
      else
      {
        for (; tid < dim; tid += 32)
        {
          // T temp = ptr[(row_index) * cache_dim + tid];
          T temp = ptr.read((row_index)*cache_dim + tid);

          // T temp = ptr.read_submit_async((row_index)*cache_dim + tid); // √

          // if (!ptr.ctx.isHit)
          // {
          //   temp = ptr.read_wait_async((row_index)*cache_dim + tid); // √
          //   ptr.ctx.isHit = true;                                    // 重要
          // }
          out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
      }
    }
    // 基于位图的CPU访问
    else
    {
      // 执行
      // printf("基于位图\n");
      if ((cpu_off & 0x1) == 1)
      {
        if (tid == 0)
          atomicAdd(d_cpu_access, 1);

        for (; tid < dim; tid += 32)
        {
          T temp = CPU_buffer.device_cpu_buffer[(cpu_off >> 1) * cache_dim + tid];
          out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
      }

      else
      {
        for (; tid < dim; tid += 32)
        {
          // T temp = ptr[(row_index) * cache_dim + tid];
          // printf("基于位图的read\n");
          T temp = ptr.read((row_index)*cache_dim + tid);
          // printf("before..start:%d, end:%d\n", ptr.start, ptr.end);

          // T temp = ptr.read_submit_async((row_index)*cache_dim + tid); // √
          // // printf("read_submit_async执行完成\n");
          // // printf("after..start:%d, end:%d\n\n", ptr.start, ptr.end);
          // // printf("ptr.ctx.isHit:%d\n", ptr.ctx.isHit);
          // if (!ptr.ctx.isHit)
          // {
          //   temp = ptr.read_wait_async((row_index)*cache_dim + tid); // √
          //   ptr.ctx.isHit = true;                                    // 重要
          //   // printf("read_wait_async执行完成\n");
          // }

          // printf("temp:%f\n", temp);
          // 输出:temp:0.032106
          // 使用 32 个线程协作读取一个特征向量(每个线程处理 dim/32 个元素)
          // printf("(bid * num_warps + warp_id) * dim + tid:%d\n", (bid * num_warps + warp_id) * dim + tid);
          out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
      }
    }
  }
}

template <typename T = float>
__global__ void clear_cache_kernel(page_cache_d_t *cache)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx == 0)
  {
    printf("Clearing cache...\n");
    clear_cache_safe(cache);
  }
    
}

template <typename T = float>
__global__ void clear_range_pages_kernel(range_d_t<T> *d_range)
{
  const uint64_t idx = threadIdx.x + static_cast<uint64_t>(blockIdx.x) * blockDim.x;
  if (idx < d_range->page_count)
  {
    d_range->pages[idx].state.store(INVALID, simt::memory_order_relaxed);
    d_range->pages[idx].offset = 0;
    d_range->pages[idx].prefetch_count.store(0, simt::memory_order_relaxed);
    d_range->pages[idx].prefetch_counter.store(0, simt::memory_order_relaxed);
    d_range->pages[idx].cpu_feature_offset = 0;
  }
}

template <typename T = float>
__global__ void set_cpu_buffer_kernel(range_d_t<T> *d_range, uint64_t *idx_ptr, int num, uint32_t pageSize)
{

  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num)
  {
    d_range->set_cpu_buffer(idx_ptr[idx], idx);
  }
}

template <typename T = float>
__global__ void set_cpu_buffer_data_kernel(array_d_t<T> *dr, T *CPU_buffer, uint64_t *idx_ptr, uint64_t dim, int num)
{
  uint64_t bid = blockIdx.x;
  bam_ptr<T> ptr(dr);
  if (bid < num)
  {
    uint64_t idx = idx_ptr[bid];

    for (uint64_t i = threadIdx.x; i < dim; i += blockDim.x)
    {
      CPU_buffer[bid * dim + i] = ptr[idx * dim + i];
    }
  }
}

template <typename T = float>
__global__ void set_window_buffering_kernel(array_d_t<T> *dr, uint64_t *index_ptr, uint64_t page_size, int hash_off)
{
  bam_ptr<T> ptr(dr);
  if (threadIdx.x == 0)
  {
    uint64_t page_idx = index_ptr[blockIdx.x] + hash_off;
    ptr.set_window_buffer_counter(page_idx * page_size / sizeof(T), 1);
  }
}

template <typename T = float>
__global__ void print_pages_ref_count_kernel(range_d_t<T> *d_range)
{
  const uint64_t idx = threadIdx.x + static_cast<uint64_t>(blockIdx.x) * blockDim.x;
  uint32_t ref_count = d_range->pages[idx].ref_count.load(simt::memory_order_acquire);
  if (idx < d_range->page_count && ref_count != 0)
  // if (idx < d_range->page_count)
  {
    // printf("page %llu ref count: %u\n", (unsigned long long)idx, d_range->pages[idx].state.load());
    printf("page %llu ref count: %u\n", (unsigned long long)idx, ref_count);
  }
}

template <typename T = float>
__global__ void print_ctx_kernel(s_ctx *d_warp_ctx)
{
  int warp_id = (threadIdx.x >> 5) + blockIdx.x * (blockDim.x / 32);
  // 每个warp的lane 0负责输出
    if ((threadIdx.x & 31) == 0 && d_warp_ctx[warp_id].addr != 0) 
    {
        
        printf("Warp %d: addr = %lu\n", warp_id, d_warp_ctx[warp_id].addr);
    }
}


__global__ void print_ctx_kernel_for(s_ctx *d_warp_ctxs, int warp_num)
{
  // 只有一个线程，直接打印
    for (uint64_t i = 0; i < warp_num; i++) {
        printf("Warp %lu: addr = %lu, page:%lu\n", i / 32, d_warp_ctxs[i].addr, d_warp_ctxs[i].page);
    }
}

template <typename T = float>
__global__ void read_kernel(array_d_t<T> *dr,
                            uint64_t num, uint64_t offset)
{
  bam_ptr<T> ptr(dr);
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (uint64_t i = 0; i < num; i++)
    {
      if (i == 0)
        printf("idx: %llu type size:%i \n", offset, (int)sizeof(T));
      // T temp = ptr[i + offset];
      printf("read data: %llu\n", (unsigned long long)ptr[i + offset]);
      // printf("float read data: %f\n", temp);
    }
  }
}

template <typename T = float>
__global__ void seq_read_kernel(array_d_t<T> *dr,
                                uint64_t num, uint64_t offset)
{
  bam_ptr<T> ptr(dr);
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (uint64_t i = 0; i < num; i++)
    {
      // if(i == 0) printf("idx: %llu type size:%i \n", offset,  (int) sizeof(T));
      T temp = ptr[i + offset];
      // printf("read data: %llu\n",  (unsigned long long) ptr[i + offset]);
      printf("read data: %f\n", (float)ptr[i + offset]);
      // printf("float read data: %f\n", temp);
    }
  }
}

template <typename T = float>
__global__ void write_feature_kernel(Controller **ctrls, page_cache_d_t *pc, array_d_t<T> *dr, T *in_tensor_ptr,
                                     uint64_t num, uint64_t page_size, uint64_t o_offset, uint64_t s_offset, uint32_t num_ctrls)
{

  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t ctrl = (tid) % (num_ctrls);
  uint64_t pc_idx = tid / num_ctrls;

  uint32_t queue = (tid) % (ctrls[ctrl]->n_qps);

  if (tid < num)
  {
    uint64_t start_block = ((o_offset + s_offset + pc_idx * page_size)) >> ctrls[ctrl]->d_qps[queue].block_size_log;

    uint64_t n_blocks = page_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
    write_data(pc, (ctrls[ctrl]->d_qps) + (queue), start_block, n_blocks, tid);
  }
}

template <typename T = float>
__global__ void write_feature_kernel2(Controller **ctrls, page_cache_d_t *pc, array_d_t<T> *dr, T *in_tensor_ptr, uint64_t dim, uint32_t num_ctrls, uint64_t offset)
{

  bam_ptr<T> ptr(dr);
  uint64_t row_index = blockIdx.x;

  for (int i = threadIdx.x; i < dim; i += blockDim.x)
  {
    ptr[(row_index)*dim + i] = in_tensor_ptr[(row_index)*dim + i + offset];
  }
}
