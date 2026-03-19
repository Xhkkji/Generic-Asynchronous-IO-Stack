

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
__global__ void read_feature_kernel_submit_async(array_d_t<T> *dr, T *out_tensor_ptr,
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
    uint64_t tid = threadIdx.x % 32;
    // printf("submit idx_idx:%d\n", idx_idx);
    s_ctx& ctx = d_warp_ctxs[idx_idx];
    if(lane_id == 0)
    {
      ctx.idx_idx = idx_idx;  // 重要，确保submit和wait使用同一个idx_idx
      ctx.row_index = row_index;  // 重要，确保submit和wait使用同一个row_index
      // printf("submit_async idx_idx:%d, row_index:%llu\n", ctx.idx_idx, (unsigned long long)ctx.row_index);
    }
    __threadfence();  // ⭐ 内存屏障：强制刷新所有内存操作到全局内存，确保其他线程能看到最新的ctx状态
    __syncwarp();
    
    // printf("warp %d 获取ctx地址:%p\n", idx_idx, &ctx);
    // printf("获取ctx完成..\n");
    // 每个线程内，分别取出自己对应负责的特征
    for (; tid < dim; tid += 32)
    {
      // T temp = ptr[(row_index) * cache_dim + tid];
      const size_t idx = (row_index)*cache_dim + tid;
      // out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = ptr[idx];
      // out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = ptr.read(idx);

      T temp = ptr.read_submit_async((row_index)*cache_dim + tid, ctx); // √
      // T temp = T(0);
      // if(lane_id == 0) printf("temp:%f\n", temp);  // √
      // 是否命中是以warp为单位的，isHIt只在warp的主线程中赋值
      __syncwarp();
      if (ctx.isHit)  // 若该warp命中，所有线程都命中，则直接写回
      {
        out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        // if(lane_id == 0) 
        // {
          // printf("submit命中,idx_idx:%d, lane_id:%d\n", idx_idx, lane_id);
        // }
        
      }
      else
      {
        out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = T(0); // 占位，后续wait时会覆盖
        // if(lane_id == 0) 
        // {
        //   printf("submit未命中,idx_idx:%d, lane_id:%d\n", idx_idx, lane_id);
        // }
        
        // ctx.isHit = true; // 重要
      }
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

  // 自加
  // uint32_t lane_id = threadIdx.x % 32;
  int lane_id = threadIdx.x % 32;

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

  if (idx_idx < num_idx)
  {
    bam_ptr<T> ptr(dr, true);

    // printf("wait idx_idx:%d\n", idx_idx);
    s_ctx& ctx = d_warp_ctxs[idx_idx];
    uint64_t row_index = index_ptr[idx_idx] + key_off;
    // uint64_t row_index = ctx.row_index;  // (不)重要，确保submit和wait使用同一个row_index
    uint64_t tid = threadIdx.x % 32;

    // Materialize all rows in the wait stage. Miss rows complete the pending IO,
    // while hit rows rebuild the cache pointer from the submit-time context and
    // overwrite any speculative submit-stage values.
    for (; tid < dim; tid += 32)
    {
      T temp = ptr.read_wait_async((row_index)*cache_dim + tid, ctx);
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
  {
    // printf("page %llu ref count: %u\n", (unsigned long long)idx, d_range->pages[idx].state.load());
    printf("page %llu ref count: %u\n", (unsigned long long)idx, ref_count);
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
