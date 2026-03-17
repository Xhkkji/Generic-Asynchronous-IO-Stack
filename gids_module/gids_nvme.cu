#include <pybind11/pybind11.h>

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stdio.h>
#include <vector>

#include <bam_nvme.h>
#include <pybind11/stl.h>
#include "gids_kernel.cu"
// #include <bafs_ptr.h>

typedef std::chrono::high_resolution_clock Clock;

namespace
{
constexpr uint64_t kWarpCtxDebugSampleCount = 0;
constexpr int64_t kAsyncDebugRows = 0;
constexpr int kAsyncDebugDims = 8;

bool env_flag_enabled(const char *name)
{
  const char *value = std::getenv(name);
  if (value == nullptr)
  {
    return false;
  }

  return value[0] != '\0' && std::strcmp(value, "0") != 0;
}

uint64_t env_u64(const char *name, uint64_t fallback)
{
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
  {
    return fallback;
  }

  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || (end != nullptr && *end != '\0'))
  {
    return fallback;
  }

  return static_cast<uint64_t>(parsed);
}

void dump_warp_ctxs(const char *stage, s_ctx *d_warp_ctxs, uint64_t total_warps)
{
  if (d_warp_ctxs == nullptr || total_warps == 0)
  {
    return;
  }

  const uint64_t requested_sample_count = env_u64("GIDS_WARP_CTX_DEBUG_SAMPLE", kWarpCtxDebugSampleCount);
  const uint64_t sample_count = total_warps < requested_sample_count ? total_warps : requested_sample_count;
  if (sample_count == 0)
  {
    return;
  }
  std::vector<unsigned char> raw(sample_count * sizeof(s_ctx));
  cuda_err_chk(cudaMemcpy(raw.data(), d_warp_ctxs, raw.size(), cudaMemcpyDeviceToHost));

  const s_ctx *ctxs = reinterpret_cast<const s_ctx *>(raw.data());
  printf("%s ctx sample (%llu / %llu warps)\n",
         stage,
         (unsigned long long)sample_count,
         (unsigned long long)total_warps);
  for (uint64_t i = 0; i < sample_count; ++i)
  {
    printf("  ctx[%llu]: idx_idx=%d row_index=%llu isHit=%d page=%llu gaddr=%llu page_trans=%u observed=%llu cid=%u ctrl=%u queue=%u base_master=%llu\n",
           (unsigned long long)i,
           ctxs[i].idx_idx,
           (unsigned long long)ctxs[i].row_index,
           ctxs[i].isHit,
           (unsigned long long)ctxs[i].page,
           (unsigned long long)ctxs[i].gaddr,
           ctxs[i].page_trans,
           (unsigned long long)ctxs[i].observed_page_translation,
           static_cast<unsigned int>(ctxs[i].cid),
           ctxs[i].ctrl,
           ctxs[i].queue,
           (unsigned long long)ctxs[i].base_master);
  }
}

template <typename TYPE>
bool debug_values_equal(TYPE lhs, TYPE rhs)
{
  return lhs == rhs;
}

template <>
bool debug_values_equal<float>(float lhs, float rhs)
{
  return std::fabs(lhs - rhs) < 1e-5f;
}

template <>
bool debug_values_equal<double>(double lhs, double rhs)
{
  return std::fabs(lhs - rhs) < 1e-8;
}

template <typename TYPE>
void debug_print_value(TYPE value)
{
  printf(" %lld", (long long)value);
}

template <>
void debug_print_value<float>(float value)
{
  printf(" %.6f", static_cast<double>(value));
}

template <>
void debug_print_value<double>(double value)
{
  printf(" %.6f", value);
}
} // namespace

void GIDS_Controllers::init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q,
                                             const std::vector<int> &ssd_list)
{
  n_ctrls = num_ctrls;
  queueDepth = q_depth;
  numQueues = num_q;
  printf("queueDepth: %llu, num_q: %llu\n", (unsigned long long)queueDepth, (unsigned long long)numQueues);

  for (size_t i = 0; i < n_ctrls; i++)
  {
    printf("SSD index: %i\n", ssd_list[i]);
    ctrls.push_back(new Controller(ctrls_paths[ssd_list[i]], nvmNamespace, cudaDevice, queueDepth, numQueues));
  }
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::cpu_backing_buffer(uint64_t dim, uint64_t len)
{
  TYPE *cpu_buffer_ptr;
  TYPE *d_cpu_buffer_ptr;

  cuda_err_chk(cudaHostAlloc((TYPE **)&cpu_buffer_ptr, sizeof(TYPE) * dim * len, cudaHostAllocMapped));
  cuda_err_chk(cudaHostGetDevicePointer((TYPE **)&d_cpu_buffer_ptr, (TYPE *)cpu_buffer_ptr, 0));

  CPU_buffer.cpu_buffer_dim = dim;
  CPU_buffer.cpu_buffer_len = len;
  CPU_buffer.cpu_buffer = cpu_buffer_ptr;
  CPU_buffer.device_cpu_buffer = d_cpu_buffer_ptr;
  cpu_buffer_flag = true;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::init_controllers(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, uint64_t cache_size, uint64_t num_ele, uint64_t num_ssd = 1)
{

  numElems = num_ele;
  read_offset = read_off;
  n_ctrls = num_ssd;
  this->pageSize = ps;
  this->dim = ps / sizeof(TYPE);
  this->total_access = 0;

  ctrls = GIDS_ctrl.ctrls;

  std::cout << "Ctrl sizes: " << ctrls.size() << std::endl;
  uint64_t page_size = pageSize;
  uint64_t n_pages = cache_size * 1024LL * 1024 / page_size;
  this->numPages = n_pages;

  std::cout << "n pages: " << (int)(this->numPages) << std::endl;
  std::cout << "page size: " << (int)(this->pageSize) << std::endl;
  std::cout << "num elements: " << this->numElems << std::endl;

  this->h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);
  page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
  uint64_t t_size = numElems * sizeof(TYPE);

  this->h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, (uint64_t)read_off,
                                    (uint64_t)(t_size / page_size), (uint64_t)0,
                                    (uint64_t)page_size, h_pc, cudaDevice,
                                    // REPLICATE
                                    STRIPE);

  this->d_range = (range_d_t<TYPE> *)h_range->d_range_ptr;

  this->vr.push_back(nullptr);
  this->vr[0] = h_range;
  this->a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);

  cudaMalloc(&d_cpu_access, sizeof(unsigned int));
  cudaMemset(d_cpu_access, 0, sizeof(unsigned));

  return;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::set_window_buffering(uint64_t id_idx, int64_t num_pages, int hash_off = 0)
{
  uint64_t *idx_ptr = (uint64_t *)id_idx;
  uint64_t page_size = pageSize;
  set_window_buffering_kernel<TYPE><<<num_pages, 32>>>(a->d_array_ptr, idx_ptr, page_size, hash_off);
  cuda_err_chk(cudaDeviceSynchronize())
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats_no_ctrl()
{
  std::cout << "print stats: ";
  this->h_pc->print_reset_stats();
  std::cout << std::endl;

  std::cout << "print array reset: ";
  this->a->print_reset_stats();
  std::cout << std::endl;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats()
{
  std::cout << "print stats: ";
  this->h_pc->print_reset_stats();
  std::cout << std::endl;

  std::cout << "print array reset: ";
  this->a->print_reset_stats();
  std::cout << std::endl;

  for (int i = 0; i < n_ctrls; i++)
  {
    std::cout << "print ctrl reset " << i << ": ";
    (this->ctrls[i])->print_reset_stats();
    std::cout << std::endl;
  }

  std::cout << "Kernel Time: \t " << this->kernel_time << std::endl;
  this->kernel_time = 0;
  std::cout << "Total Access: \t " << this->total_access << std::endl;
  this->total_access = 0;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                           int64_t num_index, int dim, int cache_dim, uint64_t key_off = 0)
{
  // printf("read_feature..\n");
  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;

  uint64_t b_size = blkSize;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();
  if (cpu_buffer_flag == false)
  {
    static bool logged_sync_mode = false;
    if (env_flag_enabled("GIDS_FORCE_SYNC_READ"))
    {
      printf("read_feature..\n");
      if (!logged_sync_mode)
      {
        printf("read_feature mode: sync baseline via read_feature_kernel\n");
        logged_sync_mode = true;
      }

      read_feature_kernel<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index, cache_dim, key_off);
      cuda_err_chk(cudaDeviceSynchronize());
    }
    else
    {
      printf("read_feature async..\n");
      s_ctx *d_warp_ctxs; // 设备端的全局warp上下文数组,一个warp持有一个上下文ctx

      // 计算总warp数
      uint64_t warps_per_block = b_size / 32;
      uint64_t total_warps = g_size * warps_per_block; // 43337 * 4 = 173348
      // 分配设备内存
      cudaMalloc(&d_warp_ctxs, total_warps * sizeof(s_ctx));
      cudaMemset(d_warp_ctxs, 0, total_warps * sizeof(s_ctx));

      read_feature_kernel_submit_async<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                                 index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
      // 等待submit完成
      cudaDeviceSynchronize();
      dump_warp_ctxs("submit", d_warp_ctxs, total_warps);

      read_feature_kernel_wait_async<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                               index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
      cuda_err_chk(cudaDeviceSynchronize());
      dump_warp_ctxs("wait", d_warp_ctxs, total_warps);

      const int64_t requested_debug_rows = static_cast<int64_t>(env_u64("GIDS_ASYNC_DEBUG_ROWS", kAsyncDebugRows));
      const int requested_debug_dims = static_cast<int>(env_u64("GIDS_ASYNC_DEBUG_DIMS", kAsyncDebugDims));
      const int64_t debug_rows = num_index < requested_debug_rows ? num_index : requested_debug_rows;
      const int debug_dims = dim < requested_debug_dims ? dim : requested_debug_dims;
      if (debug_rows > 0 && debug_dims > 0)
      {
        TYPE *ref_tensor_ptr = nullptr;
        const size_t debug_elems = static_cast<size_t>(debug_rows) * static_cast<size_t>(dim);
        cuda_err_chk(cudaMalloc(&ref_tensor_ptr, sizeof(TYPE) * debug_elems));

        const uint64_t debug_g_size = (debug_rows + n_warp - 1) / n_warp;
        read_feature_kernel<TYPE><<<debug_g_size, b_size>>>(a->d_array_ptr, ref_tensor_ptr,
                                                            index_ptr, dim, debug_rows, cache_dim, key_off);
        cuda_err_chk(cudaDeviceSynchronize());

        std::vector<TYPE> h_async(debug_elems);
        std::vector<TYPE> h_ref(debug_elems);
        cuda_err_chk(cudaMemcpy(h_async.data(), tensor_ptr, sizeof(TYPE) * debug_elems, cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemcpy(h_ref.data(), ref_tensor_ptr, sizeof(TYPE) * debug_elems, cudaMemcpyDeviceToHost));

        int mismatch_count = 0;
        std::vector<int> row_mismatch_counts(static_cast<size_t>(debug_rows), 0);
        for (size_t idx = 0; idx < debug_elems; ++idx)
        {
          if (!debug_values_equal(h_async[idx], h_ref[idx]))
          {
            ++mismatch_count;
            const size_t row = idx / static_cast<size_t>(dim);
            ++row_mismatch_counts[row];
          }
        }

        int mismatch_rows = 0;
        for (int row_mismatch_count : row_mismatch_counts)
        {
          if (row_mismatch_count != 0)
          {
            ++mismatch_rows;
          }
        }

        printf("async debug: rows=%lld dims=%d mismatches=%d/%zu mismatch_rows=%d\n",
               (long long)debug_rows,
               debug_dims,
               mismatch_count,
               debug_elems,
               mismatch_rows);
        if (mismatch_count != 0)
        {
          for (int64_t row = 0; row < debug_rows; ++row)
          {
            const int row_mismatches = row_mismatch_counts[static_cast<size_t>(row)];
            if (row_mismatches == 0)
            {
              continue;
            }

            printf("  row %lld async (mismatches=%d):",
                   (long long)row,
                   row_mismatches);
            for (int col = 0; col < debug_dims; ++col)
            {
              const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(col);
              debug_print_value(h_async[offset]);
            }
            printf("\n");

            printf("  row %lld  ref:", (long long)row);
            for (int col = 0; col < debug_dims; ++col)
            {
              const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(col);
              debug_print_value(h_ref[offset]);
            }
            printf("\n");
          }
        }

        cudaFree(ref_tensor_ptr);
      }

      // clear_cache_kernel<TYPE><<<1, 1>>>(h_pc->d_pc_ptr);
      {
        constexpr uint64_t clear_pages_block_size = 256;
        const uint64_t clear_pages_count = h_range->rdt.page_count;
        const uint64_t clear_pages_grid_size =
            (clear_pages_count + clear_pages_block_size - 1) / clear_pages_block_size;
        
        print_pages_ref_count_kernel<<<clear_pages_grid_size, clear_pages_block_size>>>(d_range);
        // clear_range_pages_kernel<TYPE><<<clear_pages_grid_size, clear_pages_block_size>>>(d_range);
      }
      // const uint64_t clear_pages_count = h_range->rdt.page_count;

      // clear_cache_kernel<TYPE><<<1, 1>>>(h_pc->d_pc_ptr);
      // {
      //   constexpr uint64_t clear_pages_block_size = 256;
      //   const uint64_t clear_pages_count = h_range->rdt.page_count;
      //   const uint64_t clear_pages_grid_size =
      //       (clear_pages_count + clear_pages_block_size - 1) / clear_pages_block_size;
      //   clear_range_pages_kernel<TYPE><<<clear_pages_grid_size, clear_pages_block_size>>>(d_range);
      // }
      

      if (d_warp_ctxs != nullptr)
        cudaFree(d_warp_ctxs);
    }
  }
  else
  {
    read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size>>>(a->d_array_ptr, d_range, tensor_ptr,
                                                                    index_ptr, dim, num_index, cache_dim, CPU_buffer, seq_flag,
                                                                    d_cpu_access, key_off);
  }
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  kernel_time += ms_fractional;
  total_access += num_index;

  return;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_hetero(int num_iter, const std::vector<uint64_t> &i_ptr_list, const std::vector<uint64_t> &i_index_ptr_list,
                                                  const std::vector<uint64_t> &num_index, int dim, int cache_dim, const std::vector<uint64_t> &key_off)
{
  // printf("read_feature..\n");
  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++)
  {
    cudaStreamCreate(&streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for (uint64_t i = 0; i < num_iter; i++)
  {
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t i_index_ptr = i_index_ptr_list[i];
    TYPE *tensor_ptr = (TYPE *)i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i] + n_warp - 1) / n_warp;

    if (cpu_buffer_flag == false)
    {
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, tensor_ptr,
                                                                   index_ptr, dim, num_index[i], cache_dim, key_off[i]);
    }
    else
    {
      // seq_flag默认为false
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, d_range, tensor_ptr,
                                                                                     index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag,
                                                                                     d_cpu_access, key_off[i]);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++)
  {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  // std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
  //         << std::endl;

  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++)
  {
    cudaStreamDestroy(streams[i]);
  }

  return;
}

// num_iter:独立的请求数
// i_ptr_list：输出张量指针列表，每个元素: GPU显存地址，指向TYPE类型的张量,含义: 输出缓冲区的设备指针数组
// 形状: 每个张量大小为 num_index[i] × dim
// i_index_ptr_list:索引指针列表
// 含义: 索引数组的设备指针数组
// 每个元素: GPU显存地址，指向int64_t类型的索引数组
// 内容: 要读取的特征行索引
// 对于第i个批次：
// i_ptr_list[i]        // 输出张量地址
// i_index_ptr_list[i]  // 输入索引地址
// num_index[i]         // 要读取的索引个数
template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged(int num_iter, const std::vector<uint64_t> &i_ptr_list, const std::vector<uint64_t> &i_index_ptr_list,
                                                  const std::vector<uint64_t> &num_index, int dim, int cache_dim = 1024)
{
  // printf("num_iter:%d\n", num_iter);
  // 同步流只需要创建一次
  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++)
  {
    cudaStreamCreate(&streams[i]);
  }

  // 在循环外部创建事件
  // cudaEvent_t events[num_iter];
  // for (int i = 0; i < num_iter; i++)
  // {
  //     cudaEventCreate(&events[i]);
  // }

  // cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for (uint64_t i = 0; i < num_iter; i++)
  {
    uint64_t i_ptr = i_ptr_list[i]; // 获取当前iter的[index_num[i], feature_dim]二维数组指针
    uint64_t i_index_ptr = i_index_ptr_list[i];
    TYPE *tensor_ptr = (TYPE *)i_ptr; // [index_num[i], feature_dim]二维数组指针
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i] + n_warp - 1) / n_warp;
    // uint64_t g_size = 128;  // 34,560(10000可运行)

    // g_size:43337, b_size:128
    printf("g_size:%d, b_size:%d\n", g_size, b_size);
    // b_size为每个block中的线程数，假设为128，则warp数量为128/32个，一个warp处理一个特征数据(1024维)

    if (cpu_buffer_flag == false)
    {
      s_ctx *d_warp_ctxs; // 设备端的全局warp上下文数组,一个warp持有一个上下文ctx

      // 计算总warp数
      uint64_t warps_per_block = b_size / 32;
      uint64_t total_warps = g_size * warps_per_block; // 43337 * 4 = 173348

      // 分配设备内存
      cudaMalloc(&d_warp_ctxs, total_warps * sizeof(s_ctx));
      cudaMemset(d_warp_ctxs, 0, total_warps * sizeof(s_ctx));

      // ========== 第1步：重置SQ/CQ队列（添加在这里！）==========
      // 在submit之前重置队列，确保这一轮从干净状态开始
      // reset_queues_for_round<<<1, 1, 0, streams[i]>>>(h_pc, i);  // 使用当前stream
      // cudaStreamSynchronize(streams[i]);  // 确保重置完成

      // read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, tensor_ptr,
      //                                                              index_ptr, dim, num_index[i], cache_dim, 0);
      read_feature_kernel_submit_async<TYPE><<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, tensor_ptr,
                                                                                index_ptr, dim, num_index[i], cache_dim, 0, d_warp_ctxs);

      // printf("read_feature_submit_async before launched..\n");
      cudaStreamSynchronize(streams[i]); // 此处被阻塞
      // 调试输出
      printf("Submit finished for iteration %lu\n", i);

      read_feature_kernel_wait_async<TYPE><<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, tensor_ptr,
                                                                              index_ptr, dim, num_index[i], cache_dim, 0, d_warp_ctxs);

      cudaStreamSynchronize(streams[i]);
      cuda_err_chk(cudaDeviceSynchronize());

      if (d_warp_ctxs != nullptr)
        cudaFree(d_warp_ctxs);
      cudaStreamDestroy(streams[i]);
      // cudaStreamDestroy(streams1[i]);
      // cudaStreamDestroy(streams2[i]);
    }
    else
    {
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, d_range, tensor_ptr,
                                                                                     index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag,
                                                                                     d_cpu_access, 0);
      // printf("read_feature_complete..\n");
    }
    total_access += num_index[i];
  }

  // for (int i = 0; i < num_iter; i++)
  // {
  //   // printf("cudaStreamSynchronize:%d\n", i); // ×
  //   cudaStreamSynchronize(streams[i]);
  // }

  // printf("cudaStreamSynchronize finished, num_iter:%d\n", num_iter); //
  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  // std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
  //         << std::endl;

  kernel_time += ms_fractional;

  // for (int i = 0; i < num_iter; i++)
  // {
  //   cudaStreamDestroy(streams[i]);
  // }
  return;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged_hetero(int num_iter, const std::vector<uint64_t> &i_ptr_list, const std::vector<uint64_t> &i_index_ptr_list,
                                                         const std::vector<uint64_t> &num_index, int dim, int cache_dim, const std::vector<uint64_t> &key_off)
{

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++)
  {
    cudaStreamCreate(&streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for (uint64_t i = 0; i < num_iter; i++)
  {
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t i_index_ptr = i_index_ptr_list[i];
    TYPE *tensor_ptr = (TYPE *)i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i] + n_warp - 1) / n_warp;

    if (cpu_buffer_flag == false)
    {
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, tensor_ptr,
                                                                   index_ptr, dim, num_index[i], cache_dim, key_off[i]);
    }
    else
    {
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, d_range, tensor_ptr,
                                                                                     index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag,
                                                                                     d_cpu_access, key_off[i]);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++)
  {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  // std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
  //         << std::endl;

  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++)
  {
    cudaStreamDestroy(streams[i]);
  }
  return;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset)
{

  //__global__ void write_feature_kernel2(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr, uint64_t dim, uint32_t num_ctrls) {
  TYPE *t_ptr = (TYPE *)tensor_ptr;
  page_cache_d_t *d_pc = (page_cache_d_t *)(h_pc->d_pc_ptr);
  size_t b_size = 128;
  printf("num of writing node data: %llu dim: %llu\n", num, dim);
  write_feature_kernel2<TYPE><<<num, b_size>>>(h_pc->pdt.d_ctrls, d_pc, a->d_array_ptr, t_ptr, dim, n_ctrls, offset / sizeof(TYPE));
  cuda_err_chk(cudaDeviceSynchronize());
  h_pc->flush_cache();
  cuda_err_chk(cudaDeviceSynchronize());
  /*
    uint64_t s_offset = 0;

    uint64_t total_cache_size = (pageSize * numPages);
    uint64_t total_tensor_size = (sizeof(TYPE) * num);
    uint64_t num_pages = total_tensor_size / pageSize;

    uint32_t n_tsteps = ceil((float)(total_tensor_size)/(float)total_cache_size);
    printf("total iter: %llu\n", (unsigned long long) n_tsteps);
    TYPE* t_ptr = (TYPE*) tensor_ptr;

    page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc -> d_pc_ptr);
    size_t b_size = 128;
    size_t g_size = (((total_tensor_size + pageSize -1) / pageSize)  + b_size - 1)/b_size;

    for (uint32_t cstep =0; cstep < n_tsteps; cstep++) {
      uint64_t cpysize = std::min(total_cache_size, (total_tensor_size-s_offset));


     // printf("first ele:%f\n", t_ptr[0]);
      cuda_err_chk(cudaMemcpy(h_pc->pdt.base_addr, t_ptr+s_offset+offset, cpysize, cudaMemcpyHostToDevice));
      printf("g size: %i num: %llu\n", g_size, num);
      write_feature_kernel<TYPE><<<100, b_size>>>(h_pc->pdt.d_ctrls, d_pc, a->d_array_ptr, t_ptr, num_pages, pageSize, offset, s_offset, n_ctrls);
      cuda_err_chk(cudaDeviceSynchronize());

    // printf("CALLLING FLUSH\n");
    // h_pc->flush_cache();
      //cuda_err_chk(cudaDeviceSynchronize());
      s_offset = s_offset + cpysize;

    }
  */
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::flush_cache()
{
  h_pc->flush_cache();
  cuda_err_chk(cudaDeviceSynchronize());
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::set_cpu_buffer(uint64_t idx_buffer, int num)
{

  int bsize = 1024;
  int grid = (num + bsize - 1) / bsize;
  uint64_t *idx_ptr = (uint64_t *)idx_buffer;
  set_cpu_buffer_kernel<TYPE><<<grid, bsize>>>(d_range, idx_ptr, num, pageSize);
  cuda_err_chk(cudaDeviceSynchronize());

  set_cpu_buffer_data_kernel<TYPE><<<num, 32>>>(a->d_array_ptr, CPU_buffer.device_cpu_buffer, idx_ptr, dim, num);
  cuda_err_chk(cudaDeviceSynchronize());

  seq_flag = false;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off)
{

  offset_array = new uint64_t[3];
  printf("set offset: in_off: %llu index_off: %llu data_off: %llu offset_ptr:%llu\n", in_off, index_off, data_off, (uint64_t)offset_array);

  offset_array[0] = (in_off);
  offset_array[1] = (index_off);
  offset_array[2] = (data_off);
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_offset_array()
{
  return ((uint64_t)offset_array);
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_array_ptr()
{
  return ((uint64_t)(a->d_array_ptr));
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_tensor(uint64_t num, uint64_t offset)
{
  printf("offset:%llu\n", (unsigned long long)offset);
  seq_read_kernel<TYPE><<<1, 1>>>(a->d_array_ptr, num, offset);
  cuda_err_chk(cudaDeviceSynchronize());
}

template <typename TYPE>
unsigned int BAM_Feature_Store<TYPE>::get_cpu_access_count()
{
  return cpu_access_count;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::flush_cpu_access_count()
{
  cpu_access_count = 0;
  cudaMemset(d_cpu_access, 0, sizeof(unsigned));
}

template <typename T>
BAM_Feature_Store<T> create_BAM_Feature_Store()
{
  return BAM_Feature_Store<T>();
}

PYBIND11_MODULE(BAM_Feature_Store, m)
{
  m.doc() = "Python bindings for an example library";

  namespace py = pybind11;

  // py::class_<BAM_Feature_Store<>, std::unique_ptr<BAM_Feature_Store<float>, py::nodelete>>(m, "BAM_Feature_Store")
  py::class_<BAM_Feature_Store<float>>(m, "BAM_Feature_Store_float")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<float>::init_controllers)
      .def("read_feature", &BAM_Feature_Store<float>::read_feature)
      .def("read_feature_hetero", &BAM_Feature_Store<float>::read_feature_hetero)

      .def("read_feature_merged_hetero", &BAM_Feature_Store<float>::read_feature_merged_hetero)
      .def("read_feature_merged", &BAM_Feature_Store<float>::read_feature_merged)
      .def("set_window_buffering", &BAM_Feature_Store<float>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<float>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<float>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<float>::flush_cache)
      .def("store_tensor", &BAM_Feature_Store<float>::store_tensor)
      .def("read_tensor", &BAM_Feature_Store<float>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<float>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<float>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<float>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<float>::get_cpu_access_count)
      .def("flush_cpu_access_count", &BAM_Feature_Store<float>::flush_cpu_access_count)

      .def("print_stats", &BAM_Feature_Store<float>::print_stats);

  py::class_<BAM_Feature_Store<int64_t>>(m, "BAM_Feature_Store_long")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<int64_t>::init_controllers)
      .def("read_feature", &BAM_Feature_Store<int64_t>::read_feature)
      .def("read_feature_hetero", &BAM_Feature_Store<int64_t>::read_feature_hetero)

      .def("read_feature_merged", &BAM_Feature_Store<int64_t>::read_feature_merged)
      .def("read_feature_merged_hetero", &BAM_Feature_Store<int64_t>::read_feature_merged_hetero)

      .def("set_window_buffering", &BAM_Feature_Store<int64_t>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<int64_t>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<int64_t>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<int64_t>::flush_cache)
      .def("store_tensor", &BAM_Feature_Store<int64_t>::store_tensor)
      .def("read_tensor", &BAM_Feature_Store<int64_t>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<int64_t>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<int64_t>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<int64_t>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<int64_t>::get_cpu_access_count)
      .def("flush_cpu_access_count", &BAM_Feature_Store<int64_t>::flush_cpu_access_count)

      .def("print_stats", &BAM_Feature_Store<int64_t>::print_stats);

  py::class_<GIDS_Controllers>(m, "GIDS_Controllers")
      .def(py::init<>())
      .def("init_GIDS_controllers", &GIDS_Controllers::init_GIDS_controllers);
}

// gids
