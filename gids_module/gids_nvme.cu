#include <pybind11/pybind11.h>

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <stdio.h>
#include <vector>

#include <bam_nvme.h>
#include <pybind11/stl.h>
#include "gids_kernel.cu"
#include <bam_iostack.cuh>
// #include <bafs_ptr.h>

typedef std::chrono::high_resolution_clock Clock;

namespace
{
constexpr uint64_t kWarpCtxDebugSampleCount = 0;
constexpr int64_t kAsyncDebugRows = 0;
constexpr int kAsyncDebugDims = 8;
// Run more CQ-service rounds before each front ready-check so the hot
// request-level aggregation happens less frequently.
constexpr uint32_t kRegisteredFrontServiceBursts = 4;
constexpr uint32_t kRegisteredWindowServiceBursts = 3;

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

template <typename TYPE>
std::string registered_queue_snapshot(const BaM_IOStack<TYPE> &iostack,
                                      uint64_t max_entries = 4)
{
  std::ostringstream oss;
  const uint64_t outstanding = iostack.outstanding_count();
  oss << "outstanding=" << outstanding;
  oss << " front_ready_request_id=" << iostack.front_ready_request_id();
  oss << " last_consumed_request_id=" << iostack.get_last_consumed_request_id();
  oss << " entries=[";
  const uint64_t sample_count = outstanding < max_entries ? outstanding : max_entries;
  for (uint64_t i = 0; i < sample_count; ++i)
  {
    const auto *meta = iostack.outstanding_at(i);
    if (meta == nullptr)
    {
      break;
    }
    if (i != 0)
    {
      oss << "; ";
    }
    oss << "{offset=" << i
        << ", request_id=" << meta->request_id
        << ", state=" << static_cast<uint32_t>(meta->state)
        << ", num_index=" << meta->num_index
        << ", dim=" << meta->dim
        << ", cache_dim=" << meta->cache_dim
        << ", ctx_stride=" << meta->ctx_stride
        << ", ctx_count=" << meta->ctx_count
        << ", index_ptr=0x" << std::hex << meta->index_ptr << std::dec
        << "}";
  }
  oss << "]";
  return oss.str();
}

template <typename TYPE>
void wait_for_registered_poll_kernel(BaM_IOStack<TYPE> &iostack,
                                     const typename BaM_IOStack<TYPE>::outstanding_meta_t &meta,
                                     uint64_t request_id,
                                     bool debug_poll)
{
  const uint64_t timeout_sec = env_u64("CIDS_REGISTERED_POLL_TIMEOUT_SEC", 0);
  if (timeout_sec == 0)
  {
    cuda_err_chk(cudaDeviceSynchronize());
    return;
  }

  cudaEvent_t done_event = nullptr;
  cuda_err_chk(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming));
  cuda_err_chk(cudaEventRecord(done_event));

  const auto start = Clock::now();
  int64_t last_logged_sec = -1;
  while (true)
  {
    const cudaError_t query_err = cudaEventQuery(done_event);
    if (query_err == cudaSuccess)
    {
      cuda_err_chk(cudaEventDestroy(done_event));
      return;
    }
    if (query_err != cudaErrorNotReady)
    {
      const std::string snapshot = registered_queue_snapshot(iostack);
      cudaEventDestroy(done_event);
      std::ostringstream oss;
      oss << "registered poll kernel query failed request_id=" << request_id
          << " cuda_error=" << static_cast<int>(query_err)
          << " (" << cudaGetErrorString(query_err) << ") "
          << snapshot;
      throw std::runtime_error(oss.str());
    }

    const auto now = Clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
    if (elapsed >= static_cast<int64_t>(timeout_sec))
    {
      const std::string snapshot = registered_queue_snapshot(iostack);
      cudaEventDestroy(done_event);
      std::ostringstream oss;
      oss << "registered poll kernel timeout request_id=" << request_id
          << " elapsed_sec=" << elapsed
          << " num_index=" << meta.num_index
          << " dim=" << meta.dim
          << " cache_dim=" << meta.cache_dim
          << " ctx_stride=" << meta.ctx_stride
          << " ctx_count=" << meta.ctx_count
          << " index_ptr=0x" << std::hex << meta.index_ptr << std::dec
          << " " << snapshot;
      throw std::runtime_error(oss.str());
    }

    if (debug_poll && elapsed > 0 && (elapsed % 5) == 0 && elapsed != last_logged_sec)
    {
      last_logged_sec = elapsed;
      printf("[REGISTERED_POLL_COMPAT] waiting request_id=%llu elapsed_sec=%lld\n",
             (unsigned long long)request_id,
             (long long)elapsed);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
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
void BAM_Feature_Store<TYPE>::enable_bam_policy_cache(uint64_t num_pages)
{
  enable_bam_policy_cache_grouped(num_pages, 1);
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::enable_bam_policy_cache_grouped(uint64_t num_policy_slots, uint64_t group_pages)
{
  if (h_pc == nullptr)
  {
    throw std::runtime_error("enable_bam_policy_cache_grouped 需要先完成 init_controllers");
  }
  if (num_policy_slots == 0)
  {
    throw std::runtime_error("enable_bam_policy_cache_grouped 需要正数 num_policy_slots");
  }
  if (group_pages == 0)
  {
    throw std::runtime_error("enable_bam_policy_cache_grouped 需要正数 group_pages");
  }
  h_pc->enable_policy_cache_grouped(num_policy_slots, group_pages);
  bam_policy_enabled = true;
  bam_policy_pages = num_policy_slots;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::update_bam_policy_scores(const std::vector<uint64_t> &page_ids, const std::vector<float> &scores)
{
  if (!bam_policy_enabled || h_pc == nullptr)
  {
    return;
  }
  if (page_ids.size() != scores.size())
  {
    throw std::runtime_error("update_bam_policy_scores 需要等长的 page_ids 和 scores");
  }
  for (size_t i = 0; i < page_ids.size(); ++i)
  {
    const uint64_t page_id = page_ids[i];
    if (page_id >= bam_policy_pages)
    {
      continue;
    }
    h_pc->update_policy_score(page_id, scores[i]);
  }
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::update_bam_policy_scores_device(uint64_t scores_ptr, uint64_t num_pages)
{
  if (!bam_policy_enabled || h_pc == nullptr)
  {
    return;
  }
  const uint64_t copy_pages = num_pages < bam_policy_pages ? num_pages : bam_policy_pages;
  if (copy_pages == 0)
  {
    return;
  }
  float *src_scores = reinterpret_cast<float *>(scores_ptr);
  cuda_err_chk(cudaMemcpy(
      h_pc->pdt.policy_scores,
      src_scores,
      copy_pages * sizeof(float),
      cudaMemcpyDeviceToDevice));
}

template <typename TYPE>
__global__ void query_sample_residency_kernel(
    range_d_t<TYPE> *range,
    const int64_t *sample_ids,
    uint8_t *out_resident,
    uint64_t num_samples,
    uint64_t pages_per_sample)
{
  const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_samples)
  {
    return;
  }

  const uint64_t sample_id = static_cast<uint64_t>(sample_ids[idx]);
  const uint64_t base_page = sample_id * pages_per_sample;
  bool resident = true;

  // 训练调度功能：
  // - 把一个 sample 视为一组连续 row/page
  // - 只有整组都 valid，才把这个 sample 当成 resident
  for (uint64_t page_idx = 0; page_idx < pages_per_sample; ++page_idx)
  {
    const uint64_t read_state =
        range->pages[base_page + page_idx].state.load(simt::memory_order_acquire);
    const uint64_t state = (read_state >> (CNT_SHIFT + 1)) & 0x03;
    if (state != V_NB)
    {
      resident = false;
      break;
    }
  }

  out_resident[idx] = resident ? 1 : 0;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::query_sample_residency_device(
    uint64_t sample_ids_ptr,
    uint64_t num_samples,
    uint64_t pages_per_sample,
    uint64_t out_ptr)
{
  if (d_range == nullptr || num_samples == 0 || pages_per_sample == 0)
  {
    return;
  }

  const auto *sample_ids = reinterpret_cast<const int64_t *>(sample_ids_ptr);
  auto *out_resident = reinterpret_cast<uint8_t *>(out_ptr);
  constexpr uint64_t kBlockSize = 256;
  const uint64_t grid_size = (num_samples + kBlockSize - 1) / kBlockSize;

  query_sample_residency_kernel<<<grid_size, kBlockSize>>>(
      d_range,
      sample_ids,
      out_resident,
      num_samples,
      pages_per_sample);
  cuda_err_chk(cudaDeviceSynchronize());
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

  auto t1 = Clock::now();
  if (cpu_buffer_flag == false)
  {
    static bool logged_sync_mode = false;
    // sync 读取功能：
    // - read_feature 固定代表同步读取
    // - async/registered 读取走独立的 submit/poll/get 接口
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
void BAM_Feature_Store<TYPE>::read_feature_submit_async(uint64_t i_index_ptr,
                                           int64_t num_index, int dim, int cache_dim, uint64_t key_off = 0)
{
  // printf("BAM_Feature_Store::read_feature_submit_async..\n");
  // TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;
  uint64_t b_size = blkSize;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  s_ctx *d_warp_ctxs; // 设备端的全局warp上下文数组,一个warp持有一个上下文ctx

  // 每个样本保留 32 份 ctx，供单线程轮询后按 warp 回填特征使用。
  uint64_t warps_per_block = b_size / 32;
  uint64_t total_warps = g_size * warps_per_block;
  uint64_t total_ctxs = total_warps * 32;
  // 分配设备内存
  cudaMalloc(&d_warp_ctxs, total_ctxs * sizeof(s_ctx));
  cudaMemset(d_warp_ctxs, 0, total_ctxs * sizeof(s_ctx));
  this->iostack.push_ctxs(d_warp_ctxs, total_ctxs);
  // this->iostack.d_warp_ctxs_array_size[0] = static_cast<uint64_t>(total_warps);
  // 无需接受返回值，只提交IO请求即可
  read_feature_kernel_submit_async<TYPE><<<g_size, b_size>>>(a->d_array_ptr,
                                                             index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  // 等待submit完成
  // cudaDeviceSynchronize();
  dump_warp_ctxs("submit", d_warp_ctxs, total_ctxs);
  
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
uint64_t BAM_Feature_Store<TYPE>::read_feature_submit_async_registered(uint64_t i_index_ptr,
                                           int64_t num_index, int dim, int cache_dim, uint64_t key_off)
{
  int64_t *index_ptr = (int64_t *)i_index_ptr;
  uint64_t b_size = blkSize;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  s_ctx *d_warp_ctxs;
  uint64_t warps_per_block = b_size / 32;
  uint64_t total_warps = g_size * warps_per_block;
  uint64_t total_ctxs = total_warps * 32;
  cudaMalloc(&d_warp_ctxs, total_ctxs * sizeof(s_ctx));
  cudaMemset(d_warp_ctxs, 0, total_ctxs * sizeof(s_ctx));
  const uint64_t request_id =
      this->iostack.register_outstanding(d_warp_ctxs, total_ctxs, i_index_ptr, num_index,
                                         dim, cache_dim, key_off);

  read_feature_kernel_submit_async<TYPE><<<g_size, b_size>>>(a->d_array_ptr,
                                                             index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  dump_warp_ctxs("submit_registered", d_warp_ctxs, total_ctxs);

  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  const float ms_fractional = static_cast<float>(us.count()) / 1000;

  kernel_time += ms_fractional;
  total_access += num_index;

  return request_id;
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::read_feature_submit_async_registered_rowctx(uint64_t i_index_ptr,
                                           int64_t num_index, int dim, int cache_dim, uint64_t key_off)
{
  const bool debug_poll = env_flag_enabled("CIDS_REGISTERED_POLL_DEBUG");
  int64_t *index_ptr = (int64_t *)i_index_ptr;
  constexpr uint64_t b_size = 256;
  const uint64_t n_warp = b_size / 32;
  const uint64_t g_size = (num_index + n_warp - 1) / n_warp;

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  s_ctx *d_row_ctxs;
  const uint64_t total_ctxs = static_cast<uint64_t>(num_index);
  cudaMalloc(&d_row_ctxs, total_ctxs * sizeof(s_ctx));
  cudaMemset(d_row_ctxs, 0, total_ctxs * sizeof(s_ctx));
  const uint64_t request_id =
      this->iostack.register_outstanding(d_row_ctxs, total_ctxs, i_index_ptr, num_index,
                                         dim, cache_dim, key_off, 1);
  if (debug_poll)
  {
    printf("[REGISTERED_SUBMIT_ROWCTX] request_id=%llu d_row_ctxs=%p total_ctxs=%llu index_ptr=0x%llx num_index=%lld dim=%d cache_dim=%d g_size=%llu\n",
           (unsigned long long)request_id,
           (void *)d_row_ctxs,
           (unsigned long long)total_ctxs,
           (unsigned long long)i_index_ptr,
           (long long)num_index,
           dim,
           cache_dim,
           (unsigned long long)g_size);
  }

  read_feature_kernel_submit_async_rowctx<TYPE><<<g_size, b_size>>>(
      a->d_array_ptr, index_ptr, dim, num_index, cache_dim, 0, d_row_ctxs);
  if (this->registered_total_logical_queues == 0)
  {
    for (uint32_t ctrl_idx = 0; ctrl_idx < this->n_ctrls; ++ctrl_idx)
    {
      Controller *ctrl = this->ctrls[ctrl_idx];
      if (ctrl != nullptr)
      {
        this->registered_total_logical_queues += ctrl->n_qps;
      }
    }
  }
  const uint64_t lookup_capacity =
      static_cast<uint64_t>(this->registered_total_logical_queues) * kRegisteredCidCapacity;
  if (lookup_capacity > 0 && lookup_capacity > this->registered_ctx_lookup_capacity)
  {
    if (this->d_registered_ctx_lookup != nullptr)
    {
      cuda_err_chk(cudaFree(this->d_registered_ctx_lookup));
      this->d_registered_ctx_lookup = nullptr;
    }
    cuda_err_chk(cudaMalloc(&this->d_registered_ctx_lookup, sizeof(s_ctx *) * lookup_capacity));
    cuda_err_chk(cudaMemset(this->d_registered_ctx_lookup, 0, sizeof(s_ctx *) * lookup_capacity));
    this->registered_ctx_lookup_capacity = lookup_capacity;
  }
  if (this->d_registered_ctx_lookup != nullptr && this->registered_total_logical_queues > 0)
  {
    constexpr uint32_t kLookupThreads = 256;
    const uint32_t lookup_blocks =
        (static_cast<uint32_t>(num_index) + kLookupThreads - 1) / kLookupThreads;
    register_registered_ctx_lookup_kernel<TYPE><<<lookup_blocks, kLookupThreads>>>(
        a->d_array_ptr,
        d_row_ctxs,
        num_index,
        this->n_ctrls,
        kRegisteredCidCapacity,
        this->d_registered_ctx_lookup);
  }
  dump_warp_ctxs("submit_registered_rowctx", d_row_ctxs, total_ctxs);

  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  const float ms_fractional = static_cast<float>(us.count()) / 1000;

  kernel_time += ms_fractional;
  total_access += num_index;
  if (debug_poll)
  {
    printf("[REGISTERED_SUBMIT_ROWCTX] done request_id=%llu outstanding=%llu\n",
           (unsigned long long)request_id,
           (unsigned long long)this->iostack.outstanding_count());
  }
  return request_id;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_wait_async(uint64_t i_ptr, uint64_t i_index_ptr,
                                           int64_t num_index, int dim, int cache_dim, uint64_t key_off = 0)
{
  // printf("BAM_Feature_Store::read_feature_wait_async..\n");
  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;

  uint64_t b_size = blkSize;  // 128
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;

  auto t1 = Clock::now();

  // printf("read_feature async..\n");
  s_ctx *d_warp_ctxs = this->iostack.front_ctxs(); // 取最早提交且尚未消费的一份 ctx

  // 每个样本保留 32 份 ctx，供单线程轮询后按 warp 回填特征使用。
  uint64_t warps_per_block = b_size / 32;
  uint64_t total_warps = g_size * warps_per_block;
  uint64_t total_ctxs = total_warps * 32;
  // 保证对应关系
  // assert(d_warp_ctxs != nullptr && this->iostack.d_warp_ctxs_array_size[0] == total_warps 
  //   && "d_warp_ctxs is null or size mismatch");
  
  read_feature_kernel_wait_async<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                            index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  // uint64_t poll_g_size = (num_index + n_warp - 1) / n_warp;

  // uint64_t poll_g_size = (num_index + b_size - 1) / b_size;
  // // read_feature_kernel_single_thread_poll<TYPE><<<poll_g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
  // //                                                           index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  // read_feature_kernel_single_page_single_thread_poll<TYPE><<<poll_g_size, b_size>>>(a->d_array_ptr,
  //                                                                                    index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  // cudaDeviceSynchronize();
  // // printf("单线程轮询已完成..\n");      
                                               
  // read_feature_kernel_get_feature_light<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
  //                                                                 index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  
  // constexpr uint64_t clear_pages_block_size = 256;
  // const uint64_t clear_pages_count = h_range->rdt.page_count;
  // const uint64_t clear_pages_grid_size =
  //     (clear_pages_count + clear_pages_block_size - 1) / clear_pages_block_size;
  // print_pages_ref_count_kernel<<<clear_pages_grid_size, clear_pages_block_size>>>(d_range);
  // print_ctx_kernel_for<<<1, 1>>>(d_warp_ctxs, total_warps);

  
  cuda_err_chk(cudaDeviceSynchronize());
  
  dump_warp_ctxs("wait", d_warp_ctxs, total_ctxs);

  if (d_warp_ctxs != nullptr)
    cudaFree(d_warp_ctxs);
  this->iostack.pop_ctxs();
    
  // cuda_err_chk(cudaDeviceSynchronize());
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
void BAM_Feature_Store<TYPE>::read_feature_single_page_single_thread_poll(uint64_t i_index_ptr,
                                           int64_t num_index, int dim, int cache_dim, uint64_t key_off = 0)
{
  printf("BAM_Feature_Store::read_feature_single_page_single_thread_poll..\n");
  int64_t *index_ptr = (int64_t *)i_index_ptr;
  uint64_t b_size = blkSize;  // 128
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;

  // cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  // printf("read_feature async..\n");
  s_ctx *d_warp_ctxs = this->iostack.front_ctxs(); // 取最早提交且尚未消费的一份 ctx

  // 每个样本保留 32 份 ctx，供单线程轮询后按 warp 回填特征使用。
  uint64_t warps_per_block = b_size / 32;
  uint64_t total_warps = g_size * warps_per_block;
  uint64_t total_ctxs = total_warps * 32;
  // 保证对应关系
  // assert(d_warp_ctxs != nullptr && this->iostack.d_warp_ctxs_array_size[0] == total_warps 
  //   && "d_warp_ctxs is null or size mismatch");
  
  // read_feature_kernel_wait_async<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
  //                                                           index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  // uint64_t poll_g_size = (num_index + n_warp - 1) / n_warp;
  uint64_t poll_g_size = (num_index + b_size - 1) / b_size;
  // read_feature_kernel_single_thread_poll<TYPE><<<poll_g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
  //                                                           index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  read_feature_kernel_single_page_single_thread_poll<TYPE><<<poll_g_size, b_size>>>(a->d_array_ptr,
                                                                                     index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  // printf("单线程轮询已完成..\n");      
                                               
  // read_feature_kernel_get_feature_light<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
  //                                                                 index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  
  // constexpr uint64_t clear_pages_block_size = 256;
  // const uint64_t clear_pages_count = h_range->rdt.page_count;
  // const uint64_t clear_pages_grid_size =
  //     (clear_pages_count + clear_pages_block_size - 1) / clear_pages_block_size;
  // print_pages_ref_count_kernel<<<clear_pages_grid_size, clear_pages_block_size>>>(d_range);
  // print_ctx_kernel_for<<<1, 1>>>(d_warp_ctxs, total_warps);
  dump_warp_ctxs("wait", d_warp_ctxs, total_ctxs);
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
void BAM_Feature_Store<TYPE>::read_feature_single_page_single_thread_poll_registered(uint64_t request_id)
{
  const auto *outstanding = this->iostack.front_outstanding();
  if (outstanding == nullptr)
  {
    throw std::runtime_error("No registered outstanding request to poll");
  }
  if (request_id != 0 && outstanding->request_id != request_id)
  {
    throw std::runtime_error("Registered outstanding request id mismatch");
  }

  printf("BAM_Feature_Store::read_feature_single_page_single_thread_poll_registered..\n");
  int64_t *index_ptr = reinterpret_cast<int64_t *>(outstanding->index_ptr);
  const int64_t num_index = outstanding->num_index;
  const int dim = outstanding->dim;
  const int cache_dim = outstanding->cache_dim;
  uint64_t b_size = blkSize;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;

  auto t1 = Clock::now();

  s_ctx *d_warp_ctxs = this->iostack.front_ctxs();
  uint64_t warps_per_block = b_size / 32;
  uint64_t total_warps = g_size * warps_per_block;
  uint64_t total_ctxs = total_warps * 32;
  uint64_t poll_g_size = (num_index + b_size - 1) / b_size;
  read_feature_kernel_single_page_single_thread_poll<TYPE><<<poll_g_size, b_size>>>(
      a->d_array_ptr, index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);

  dump_warp_ctxs("wait_registered", d_warp_ctxs, total_ctxs);
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  const float ms_fractional = static_cast<float>(us.count()) / 1000;

  kernel_time += ms_fractional;
  total_access += num_index;
}

template <typename TYPE>
static void poll_registered_outstanding_at(BAM_Feature_Store<TYPE> *store, uint64_t offset, uint64_t request_id)
{
  const auto *outstanding = store->iostack.outstanding_at(offset);
  if (outstanding == nullptr)
  {
    throw std::runtime_error("No registered outstanding request at offset");
  }
  if (request_id != 0 && outstanding->request_id != request_id)
  {
    throw std::runtime_error("Registered outstanding request id mismatch at offset");
  }

  int64_t *index_ptr = reinterpret_cast<int64_t *>(outstanding->index_ptr);
  const int64_t num_index = outstanding->num_index;
  const int dim = outstanding->dim;
  const int cache_dim = outstanding->cache_dim;
  uint64_t b_size = store->blkSize;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;

  auto t1 = Clock::now();

  s_ctx *d_warp_ctxs = store->iostack.ctxs_at(offset);
  if (d_warp_ctxs == nullptr)
  {
    throw std::runtime_error("Registered ctxs are missing at offset");
  }
  uint64_t warps_per_block = b_size / 32;
  uint64_t total_warps = g_size * warps_per_block;
  uint64_t total_ctxs = total_warps * 32;
  uint64_t poll_g_size = (num_index + b_size - 1) / b_size;
  read_feature_kernel_single_page_single_thread_poll<TYPE><<<poll_g_size, b_size>>>(
      store->a->d_array_ptr, index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);

  dump_warp_ctxs("wait_registered_window", d_warp_ctxs, total_ctxs);
  cudaMemcpy(&store->cpu_access_count, store->d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  const float ms_fractional = static_cast<float>(us.count()) / 1000;

  store->kernel_time += ms_fractional;
  store->total_access += num_index;
}

template <typename TYPE>
static uint32_t service_registered_completions_burst(BAM_Feature_Store<TYPE> *store, uint32_t rounds)
{
  static uint64_t debug_round = 0;
  const uint32_t request_count = static_cast<uint32_t>(store->iostack.outstanding_count());
  if (request_count == 0)
  {
    return 0;
  }

  if (store->registered_total_logical_queues == 0)
  {
    for (uint32_t ctrl_idx = 0; ctrl_idx < store->n_ctrls; ++ctrl_idx)
    {
      Controller *ctrl = store->ctrls[ctrl_idx];
      if (ctrl != nullptr)
      {
        store->registered_total_logical_queues += ctrl->n_qps;
      }
    }
  }
  if (store->registered_total_logical_queues == 0 || store->d_registered_ctx_lookup == nullptr)
  {
    return 0;
  }

  constexpr uint32_t kThreadsPerBlock = 128;
  constexpr uint32_t kMaxEventsPerQueue = 64;
  const uint32_t total_logical_queues = store->registered_total_logical_queues;
  const uint32_t blocks = (total_logical_queues + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const uint32_t effective_rounds = std::max<uint32_t>(1, rounds);
  const bool debug_enabled = env_flag_enabled("GIDS_REGISTERED_DEBUG");
  uint32_t *d_event_count = nullptr;
  uint32_t *d_progress_count = nullptr;
  uint32_t *d_lookup_miss_count = nullptr;

  if (debug_enabled)
  {
    if (store->d_registered_service_progress == nullptr)
    {
      cuda_err_chk(cudaMalloc(&store->d_registered_service_progress, sizeof(uint32_t)));
    }
    if (store->d_registered_service_events == nullptr)
    {
      cuda_err_chk(cudaMalloc(&store->d_registered_service_events, sizeof(uint32_t)));
    }
    if (store->d_registered_service_lookup_misses == nullptr)
    {
      cuda_err_chk(cudaMalloc(&store->d_registered_service_lookup_misses, sizeof(uint32_t)));
    }
    cuda_err_chk(cudaMemset(store->d_registered_service_events, 0, sizeof(uint32_t)));
    cuda_err_chk(cudaMemset(store->d_registered_service_progress, 0, sizeof(uint32_t)));
    cuda_err_chk(cudaMemset(store->d_registered_service_lookup_misses, 0, sizeof(uint32_t)));
    d_event_count = store->d_registered_service_events;
    d_progress_count = store->d_registered_service_progress;
    d_lookup_miss_count = store->d_registered_service_lookup_misses;
  }

  auto t1 = Clock::now();
  for (uint32_t round = 0; round < effective_rounds; ++round)
  {
    service_registered_cq_window_kernel<TYPE><<<blocks, kThreadsPerBlock>>>(
        store->a->d_array_ptr,
        store->n_ctrls,
        total_logical_queues,
        d_event_count,
        d_progress_count,
        d_lookup_miss_count,
        kMaxEventsPerQueue,
        kRegisteredCidCapacity,
        store->d_registered_ctx_lookup);
  }

  uint32_t event_count = 0;
  uint32_t progress_count = 0;
  uint32_t lookup_miss_count = 0;
  if (debug_enabled)
  {
    cuda_err_chk(cudaDeviceSynchronize());
    auto t2 = Clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    store->kernel_time += static_cast<float>(us.count()) / 1000.0f;

    cuda_err_chk(cudaMemcpy(&event_count, store->d_registered_service_events,
                            sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cuda_err_chk(cudaMemcpy(&progress_count, store->d_registered_service_progress,
                            sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cuda_err_chk(cudaMemcpy(&lookup_miss_count, store->d_registered_service_lookup_misses,
                            sizeof(uint32_t), cudaMemcpyDeviceToHost));

    ++debug_round;
    const bool anomaly = lookup_miss_count > 0;
    const bool early_round = debug_round <= 4;
    const bool stall = event_count == 0 && progress_count == 0 && debug_round <= 16;
    const bool periodic = (debug_round % 512) == 0;
    if (anomaly || early_round || stall || periodic)
    {
      printf("[CQ_SERVICE_ROUND] round=%llu outstanding=%u events=%u hits=%u misses=%u queues=%u\n",
             (unsigned long long)debug_round,
             request_count,
             event_count,
             progress_count,
             lookup_miss_count,
             total_logical_queues);
    }
  }
  return progress_count;
}

template <typename TYPE>
static bool registered_request_ready_at(BAM_Feature_Store<TYPE> *store, uint64_t offset)
{
  const auto *outstanding = store->iostack.outstanding_at(offset);
  if (outstanding == nullptr)
  {
    return false;
  }
  if (outstanding->state == BaM_IOStack<TYPE>::READY)
  {
    return true;
  }

  s_ctx *d_warp_ctxs = store->iostack.ctxs_at(offset);
  if (d_warp_ctxs == nullptr)
  {
    return false;
  }
  if (store->d_registered_try_pending == nullptr)
  {
    cuda_err_chk(cudaMalloc(&store->d_registered_try_pending, sizeof(uint32_t)));
  }
  cuda_err_chk(cudaMemset(store->d_registered_try_pending, 0, sizeof(uint32_t)));

  constexpr uint32_t kThreadsPerBlock = 256;
  const uint32_t ctx_stride = outstanding->ctx_stride;
  const uint32_t blocks = (static_cast<uint32_t>(outstanding->num_index) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  check_registered_request_ready_kernel<TYPE><<<blocks, kThreadsPerBlock>>>(
      store->a->d_array_ptr,
      d_warp_ctxs,
      outstanding->num_index,
      ctx_stride,
      store->d_registered_try_pending);
  cuda_err_chk(cudaDeviceSynchronize());

  uint32_t pending_count = 0;
  cuda_err_chk(cudaMemcpy(&pending_count, store->d_registered_try_pending,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
  return pending_count == 0;
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::service_registered_poll()
{
  const auto *outstanding = this->iostack.front_outstanding();
  if (outstanding == nullptr)
  {
    return 0;
  }
  if (outstanding->state == BaM_IOStack<TYPE>::READY)
  {
    return outstanding->request_id;
  }

  const uint64_t request_id = outstanding->request_id;
  read_feature_single_page_single_thread_poll_registered(request_id);
  if (!this->iostack.mark_front_ready(request_id))
  {
    throw std::runtime_error("Failed to mark registered outstanding request as ready");
  }
  return request_id;
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::service_registered_poll_compatible()
{
  const bool debug_poll = env_flag_enabled("CIDS_REGISTERED_POLL_DEBUG");
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] enter\n");
  }
  const auto *outstanding = this->iostack.front_outstanding();
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] front_outstanding=%p\n", (const void *)outstanding);
  }
  if (outstanding == nullptr)
  {
    if (debug_poll)
    {
      printf("[REGISTERED_POLL_COMPAT] no outstanding request\n");
    }
    return 0;
  }
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] about to read request_id\n");
  }
  const uint64_t request_id = outstanding->request_id;
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] request_id=%llu\n", (unsigned long long)request_id);
    printf("[REGISTERED_POLL_COMPAT] about to read state\n");
  }
  const uint32_t request_state = static_cast<uint32_t>(outstanding->state);
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] state=%u\n", request_state);
  }
  if (request_state == static_cast<uint32_t>(BaM_IOStack<TYPE>::READY))
  {
    if (debug_poll)
    {
      printf("[REGISTERED_POLL_COMPAT] already ready request_id=%llu\n",
             (unsigned long long)request_id);
    }
    return request_id;
  }
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] about to read ctx_stride\n");
  }
  const uint32_t ctx_stride = outstanding->ctx_stride;
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] ctx_stride=%u\n", ctx_stride);
  }
  if (ctx_stride != 1)
  {
    if (debug_poll)
    {
      printf("[REGISTERED_POLL_COMPAT] fallback request_id=%llu ctx_stride=%u num_index=%lld\n",
             (unsigned long long)request_id,
             ctx_stride,
             (long long)outstanding->num_index);
    }
    return service_registered_poll();
  }
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] about to read num_index/dim/cache_dim/index_ptr/front_ctxs\n");
  }
  const int64_t num_index = outstanding->num_index;
  const int dim = outstanding->dim;
  const int cache_dim = outstanding->cache_dim;
  int64_t *index_ptr = reinterpret_cast<int64_t *>(outstanding->index_ptr);
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] num_index=%lld dim=%d cache_dim=%d index_ptr=0x%llx\n",
           (long long)num_index,
           dim,
           cache_dim,
           (unsigned long long)outstanding->index_ptr);
  }
  s_ctx *d_row_ctxs = this->iostack.front_ctxs();
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] front_ctxs=%p\n", (void *)d_row_ctxs);
  }
  if (d_row_ctxs == nullptr)
  {
    throw std::runtime_error("No registered rowctx request to poll");
  }

  constexpr uint32_t b_size = 256;
  const uint32_t g_size = (static_cast<uint32_t>(num_index) + b_size - 1) / b_size;
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] before launch request_id=%llu\n",
           (unsigned long long)request_id);
    printf("[REGISTERED_POLL_COMPAT] launch request_id=%llu num_index=%lld dim=%d cache_dim=%d g_size=%u b_size=%u\n",
           (unsigned long long)request_id,
           (long long)num_index,
           dim,
           cache_dim,
           g_size,
           b_size);
  }
  read_feature_kernel_single_page_single_thread_poll_rowctx<TYPE><<<g_size, b_size>>>(
      a->d_array_ptr, index_ptr, dim, num_index, cache_dim, 0, d_row_ctxs);
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] launched request_id=%llu\n",
           (unsigned long long)request_id);
  }
  cudaError_t launch_err = cudaPeekAtLastError();
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] post-launch cudaPeekAtLastError=%d (%s)\n",
           static_cast<int>(launch_err),
           cudaGetErrorString(launch_err));
  }
  wait_for_registered_poll_kernel(this->iostack, *outstanding, request_id, debug_poll);
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] synchronized request_id=%llu\n",
           (unsigned long long)request_id);
  }

  if (!this->iostack.mark_front_ready(request_id))
  {
    throw std::runtime_error("Failed to mark registered rowctx request as ready");
  }
  if (debug_poll)
  {
    printf("[REGISTERED_POLL_COMPAT] marked ready request_id=%llu\n",
           (unsigned long long)request_id);
  }
  return request_id;
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::service_registered_try_poll()
{
  const auto *outstanding = this->iostack.front_outstanding();
  if (outstanding == nullptr)
  {
    return 0;
  }
  if (outstanding->state == BaM_IOStack<TYPE>::READY)
  {
    return outstanding->request_id;
  }

  const uint64_t request_id = outstanding->request_id;
  service_registered_completions_burst(this, kRegisteredFrontServiceBursts);
  if (registered_request_ready_at(this, 0))
  {
    if (!this->iostack.mark_front_ready(request_id))
    {
      throw std::runtime_error("Failed to mark registered outstanding request as ready");
    }
    return request_id;
  }
  return this->iostack.front_ready_request_id();
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::service_registered_try_poll_window_skip_front(uint64_t window_size)
{
  const uint64_t total = this->iostack.outstanding_count();
  if (total <= 1)
  {
    return this->iostack.front_ready_request_id();
  }

  service_registered_completions_burst(this, kRegisteredWindowServiceBursts);

  const uint64_t limit = std::min<uint64_t>(std::max<uint64_t>(1, window_size), total - 1);
  for (uint64_t inner_offset = 0; inner_offset < limit; ++inner_offset)
  {
    const uint64_t offset = inner_offset + 1;
    const auto *outstanding = this->iostack.outstanding_at(offset);
    if (outstanding == nullptr)
    {
      break;
    }
    if (outstanding->state != BaM_IOStack<TYPE>::SUBMITTED)
    {
      continue;
    }

    const uint64_t request_id = outstanding->request_id;
    if (registered_request_ready_at(this, offset))
    {
      if (!this->iostack.mark_ready_at(offset, request_id))
      {
        throw std::runtime_error("Failed to mark registered outstanding request as ready in skip-front try window");
      }
    }
  }

  return this->iostack.front_ready_request_id();
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::read_feature_get_feature_light_registered(uint64_t i_ptr)
{
  const auto *outstanding = this->iostack.front_outstanding();
  if (outstanding == nullptr)
  {
    throw std::runtime_error("No registered outstanding request to get");
  }
  if (outstanding->state != BaM_IOStack<TYPE>::READY)
  {
    throw std::runtime_error("Registered outstanding request is not ready for get");
  }

  const uint64_t request_id = outstanding->request_id;
  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)outstanding->index_ptr;
  const int64_t num_index = outstanding->num_index;
  const int dim = outstanding->dim;
  const int cache_dim = outstanding->cache_dim;
  uint64_t b_size = blkSize;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;
  auto t1 = Clock::now();

  s_ctx *d_warp_ctxs = this->iostack.front_ctxs();
  uint64_t warps_per_block = b_size / 32;
  uint64_t total_warps = g_size * warps_per_block;
  uint64_t total_ctxs = total_warps * 32;

  read_feature_kernel_get_feature_light<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                                  index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  dump_warp_ctxs("wait_registered_get", d_warp_ctxs, total_ctxs);

  if (!this->iostack.mark_front_consumed(request_id))
  {
    throw std::runtime_error("Failed to mark registered outstanding request as consumed");
  }
  if (d_warp_ctxs != nullptr)
    cudaFree(d_warp_ctxs);
  this->iostack.pop_ctxs();

  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  const float ms_fractional = static_cast<float>(us.count()) / 1000;

  kernel_time += ms_fractional;
  total_access += num_index;
  return request_id;
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::read_feature_get_feature_light_registered_rowctx(uint64_t i_ptr)
{
  const auto *outstanding = this->iostack.front_outstanding();
  if (outstanding == nullptr)
  {
    throw std::runtime_error("No registered outstanding request to get");
  }
  if (outstanding->state != BaM_IOStack<TYPE>::READY)
  {
    throw std::runtime_error("Registered outstanding request is not ready for get");
  }

  const uint64_t request_id = outstanding->request_id;
  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)outstanding->index_ptr;
  const int64_t num_index = outstanding->num_index;
  const int dim = outstanding->dim;
  const int cache_dim = outstanding->cache_dim;
  constexpr uint64_t b_size = 128;
  const uint64_t n_warp = b_size / 32;
  const uint64_t g_size = (num_index + n_warp - 1) / n_warp;
  auto t1 = Clock::now();

  s_ctx *d_row_ctxs = this->iostack.front_ctxs();
  const uint64_t total_ctxs = static_cast<uint64_t>(num_index);

  read_feature_kernel_get_feature_light_rowctx<TYPE><<<g_size, b_size>>>(
      a->d_array_ptr, tensor_ptr, index_ptr, dim, num_index, cache_dim, 0, d_row_ctxs);
  dump_warp_ctxs("wait_registered_get_rowctx", d_row_ctxs, total_ctxs);

  if (!this->iostack.mark_front_consumed(request_id))
  {
    throw std::runtime_error("Failed to mark registered outstanding request as consumed");
  }
  if (d_row_ctxs != nullptr)
    cudaFree(d_row_ctxs);
  this->iostack.pop_ctxs();

  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  const float ms_fractional = static_cast<float>(us.count()) / 1000;

  kernel_time += ms_fractional;
  total_access += num_index;
  return request_id;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_get_feature_light(uint64_t i_ptr, uint64_t i_index_ptr,
                                           int64_t num_index, int dim, int cache_dim, uint64_t key_off = 0)
{
  printf("BAM_Feature_Store::read_feature_get_feature_light..\n");
  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;
  uint64_t b_size = blkSize;  // 128
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;
  auto t1 = Clock::now();

  s_ctx *d_warp_ctxs = this->iostack.front_ctxs(); // 取最早提交且尚未消费的一份 ctx

  // 每个样本保留 32 份 ctx，供单线程轮询后按 warp 回填特征使用。
  uint64_t warps_per_block = b_size / 32;
  uint64_t total_warps = g_size * warps_per_block;
  uint64_t total_ctxs = total_warps * 32;
  // 保证对应关系
  // assert(d_warp_ctxs != nullptr && this->iostack.d_warp_ctxs_array_size[0] == total_warps 
  //   && "d_warp_ctxs is null or size mismatch");
                                       
  read_feature_kernel_get_feature_light<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                                  index_ptr, dim, num_index, cache_dim, 0, d_warp_ctxs);
  
  // constexpr uint64_t clear_pages_block_size = 256;
  // const uint64_t clear_pages_count = h_range->rdt.page_count;
  // const uint64_t clear_pages_grid_size =
  //     (clear_pages_count + clear_pages_block_size - 1) / clear_pages_block_size;
  // print_pages_ref_count_kernel<<<clear_pages_grid_size, clear_pages_block_size>>>(d_range);
  // print_ctx_kernel_for<<<1, 1>>>(d_warp_ctxs, total_warps);

  dump_warp_ctxs("wait", d_warp_ctxs, total_ctxs);
  if (d_warp_ctxs != nullptr)
    cudaFree(d_warp_ctxs);
  this->iostack.pop_ctxs();
  
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
uint64_t BAM_Feature_Store<TYPE>::get_registered_outstanding_count() const
{
  return this->iostack.outstanding_count();
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_registered_front_request_id() const
{
  const auto *outstanding = this->iostack.front_outstanding();
  return outstanding == nullptr ? 0 : outstanding->request_id;
}

template <typename TYPE>
bool BAM_Feature_Store<TYPE>::registered_front_ready() const
{
  return this->iostack.front_outstanding_ready();
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_registered_ready_front_request_id() const
{
  return this->iostack.front_ready_request_id();
}

template <typename TYPE>
uint32_t BAM_Feature_Store<TYPE>::get_registered_front_state() const
{
  return static_cast<uint32_t>(this->iostack.front_state());
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_registered_request_id_at(uint64_t offset) const
{
  const auto *outstanding = this->iostack.outstanding_at(offset);
  return outstanding == nullptr ? 0 : outstanding->request_id;
}

template <typename TYPE>
uint32_t BAM_Feature_Store<TYPE>::get_registered_request_state_at(uint64_t offset) const
{
  const auto *outstanding = this->iostack.outstanding_at(offset);
  if (outstanding == nullptr)
  {
    return static_cast<uint32_t>(BaM_IOStack<TYPE>::CONSUMED);
  }
  return static_cast<uint32_t>(outstanding->state);
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_registered_last_consumed_request_id() const
{
  return this->iostack.get_last_consumed_request_id();
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
      read_feature_kernel_submit_async<TYPE><<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr,
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
      // 异步函数绑定
      .def("read_feature_submit_async", &BAM_Feature_Store<float>::read_feature_submit_async)
      .def("read_feature_submit_async_registered", &BAM_Feature_Store<float>::read_feature_submit_async_registered)
      .def("read_feature_submit_async_registered_rowctx", &BAM_Feature_Store<float>::read_feature_submit_async_registered_rowctx)
      .def("read_feature_wait_async", &BAM_Feature_Store<float>::read_feature_wait_async)
      .def("read_feature_single_page_single_thread_poll", &BAM_Feature_Store<float>::read_feature_single_page_single_thread_poll)
      .def("service_registered_poll", &BAM_Feature_Store<float>::service_registered_poll)
      .def("service_registered_poll_compatible", &BAM_Feature_Store<float>::service_registered_poll_compatible)
      .def("service_registered_try_poll", &BAM_Feature_Store<float>::service_registered_try_poll)
      .def("service_registered_try_poll_window_skip_front", &BAM_Feature_Store<float>::service_registered_try_poll_window_skip_front)
      .def("read_feature_get_feature_light", &BAM_Feature_Store<float>::read_feature_get_feature_light)
      .def("read_feature_get_feature_light_registered", &BAM_Feature_Store<float>::read_feature_get_feature_light_registered)
      .def("read_feature_get_feature_light_registered_rowctx", &BAM_Feature_Store<float>::read_feature_get_feature_light_registered_rowctx)
      .def("get_registered_outstanding_count", &BAM_Feature_Store<float>::get_registered_outstanding_count)
      .def("get_registered_front_request_id", &BAM_Feature_Store<float>::get_registered_front_request_id)
      .def("registered_front_ready", &BAM_Feature_Store<float>::registered_front_ready)
      .def("get_registered_ready_front_request_id", &BAM_Feature_Store<float>::get_registered_ready_front_request_id)
      .def("get_registered_front_state", &BAM_Feature_Store<float>::get_registered_front_state)
      .def("get_registered_request_id_at", &BAM_Feature_Store<float>::get_registered_request_id_at)
      .def("get_registered_request_state_at", &BAM_Feature_Store<float>::get_registered_request_state_at)
      .def("get_registered_last_consumed_request_id", &BAM_Feature_Store<float>::get_registered_last_consumed_request_id)

      .def("read_feature_hetero", &BAM_Feature_Store<float>::read_feature_hetero)
      .def("read_feature_merged_hetero", &BAM_Feature_Store<float>::read_feature_merged_hetero)
      .def("read_feature_merged", &BAM_Feature_Store<float>::read_feature_merged)
      .def("set_window_buffering", &BAM_Feature_Store<float>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<float>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<float>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<float>::flush_cache)
      .def("enable_bam_policy_cache", &BAM_Feature_Store<float>::enable_bam_policy_cache)
      .def("enable_bam_policy_cache_grouped", &BAM_Feature_Store<float>::enable_bam_policy_cache_grouped)
      .def("update_bam_policy_scores", &BAM_Feature_Store<float>::update_bam_policy_scores)
      .def("update_bam_policy_scores_device", &BAM_Feature_Store<float>::update_bam_policy_scores_device)
      .def("query_sample_residency_device", &BAM_Feature_Store<float>::query_sample_residency_device)
      .def("store_tensor", &BAM_Feature_Store<float>::store_tensor)
      .def("read_tensor", &BAM_Feature_Store<float>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<float>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<float>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<float>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<float>::get_cpu_access_count)
      .def("flush_cpu_access_count", &BAM_Feature_Store<float>::flush_cpu_access_count)

      .def("print_stats", &BAM_Feature_Store<float>::print_stats);

  py::class_<BAM_Feature_Store<uint8_t>>(m, "BAM_Feature_Store_byte")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<uint8_t>::init_controllers)
      .def("read_feature", &BAM_Feature_Store<uint8_t>::read_feature)
      // 异步函数绑定
      .def("read_feature_submit_async", &BAM_Feature_Store<uint8_t>::read_feature_submit_async)
      .def("read_feature_submit_async_registered", &BAM_Feature_Store<uint8_t>::read_feature_submit_async_registered)
      .def("read_feature_submit_async_registered_rowctx", &BAM_Feature_Store<uint8_t>::read_feature_submit_async_registered_rowctx)
      .def("read_feature_wait_async", &BAM_Feature_Store<uint8_t>::read_feature_wait_async)
      .def("read_feature_single_page_single_thread_poll", &BAM_Feature_Store<uint8_t>::read_feature_single_page_single_thread_poll)
      .def("service_registered_poll", &BAM_Feature_Store<uint8_t>::service_registered_poll)
      .def("service_registered_poll_compatible", &BAM_Feature_Store<uint8_t>::service_registered_poll_compatible)
      .def("service_registered_try_poll", &BAM_Feature_Store<uint8_t>::service_registered_try_poll)
      .def("service_registered_try_poll_window_skip_front", &BAM_Feature_Store<uint8_t>::service_registered_try_poll_window_skip_front)
      .def("read_feature_get_feature_light", &BAM_Feature_Store<uint8_t>::read_feature_get_feature_light)
      .def("read_feature_get_feature_light_registered", &BAM_Feature_Store<uint8_t>::read_feature_get_feature_light_registered)
      .def("read_feature_get_feature_light_registered_rowctx", &BAM_Feature_Store<uint8_t>::read_feature_get_feature_light_registered_rowctx)
      .def("get_registered_outstanding_count", &BAM_Feature_Store<uint8_t>::get_registered_outstanding_count)
      .def("get_registered_front_request_id", &BAM_Feature_Store<uint8_t>::get_registered_front_request_id)
      .def("registered_front_ready", &BAM_Feature_Store<uint8_t>::registered_front_ready)
      .def("get_registered_ready_front_request_id", &BAM_Feature_Store<uint8_t>::get_registered_ready_front_request_id)
      .def("get_registered_front_state", &BAM_Feature_Store<uint8_t>::get_registered_front_state)
      .def("get_registered_request_id_at", &BAM_Feature_Store<uint8_t>::get_registered_request_id_at)
      .def("get_registered_request_state_at", &BAM_Feature_Store<uint8_t>::get_registered_request_state_at)
      .def("get_registered_last_consumed_request_id", &BAM_Feature_Store<uint8_t>::get_registered_last_consumed_request_id)

      .def("read_feature_hetero", &BAM_Feature_Store<uint8_t>::read_feature_hetero)
      .def("read_feature_merged_hetero", &BAM_Feature_Store<uint8_t>::read_feature_merged_hetero)
      .def("read_feature_merged", &BAM_Feature_Store<uint8_t>::read_feature_merged)
      .def("set_window_buffering", &BAM_Feature_Store<uint8_t>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<uint8_t>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<uint8_t>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<uint8_t>::flush_cache)
      .def("enable_bam_policy_cache", &BAM_Feature_Store<uint8_t>::enable_bam_policy_cache)
      .def("enable_bam_policy_cache_grouped", &BAM_Feature_Store<uint8_t>::enable_bam_policy_cache_grouped)
      .def("update_bam_policy_scores", &BAM_Feature_Store<uint8_t>::update_bam_policy_scores)
      .def("update_bam_policy_scores_device", &BAM_Feature_Store<uint8_t>::update_bam_policy_scores_device)
      .def("query_sample_residency_device", &BAM_Feature_Store<uint8_t>::query_sample_residency_device)
      .def("store_tensor", &BAM_Feature_Store<uint8_t>::store_tensor)
      .def("read_tensor", &BAM_Feature_Store<uint8_t>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<uint8_t>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<uint8_t>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<uint8_t>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<uint8_t>::get_cpu_access_count)
      .def("flush_cpu_access_count", &BAM_Feature_Store<uint8_t>::flush_cpu_access_count)

      .def("print_stats", &BAM_Feature_Store<uint8_t>::print_stats);

  py::class_<BAM_Feature_Store<int64_t>>(m, "BAM_Feature_Store_long")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<int64_t>::init_controllers)
      .def("read_feature", &BAM_Feature_Store<int64_t>::read_feature)
      .def("read_feature_hetero", &BAM_Feature_Store<int64_t>::read_feature_hetero)
      // 异步函数绑定
      .def("read_feature_submit_async", &BAM_Feature_Store<int64_t>::read_feature_submit_async)
      .def("read_feature_submit_async_registered", &BAM_Feature_Store<int64_t>::read_feature_submit_async_registered)
      .def("read_feature_submit_async_registered_rowctx", &BAM_Feature_Store<int64_t>::read_feature_submit_async_registered_rowctx)
      .def("read_feature_wait_async", &BAM_Feature_Store<int64_t>::read_feature_wait_async)
      .def("service_registered_poll", &BAM_Feature_Store<int64_t>::service_registered_poll)
      .def("service_registered_poll_compatible", &BAM_Feature_Store<int64_t>::service_registered_poll_compatible)
      .def("service_registered_try_poll", &BAM_Feature_Store<int64_t>::service_registered_try_poll)
      .def("service_registered_try_poll_window_skip_front", &BAM_Feature_Store<int64_t>::service_registered_try_poll_window_skip_front)
      .def("read_feature_get_feature_light_registered", &BAM_Feature_Store<int64_t>::read_feature_get_feature_light_registered)
      .def("read_feature_get_feature_light_registered_rowctx", &BAM_Feature_Store<int64_t>::read_feature_get_feature_light_registered_rowctx)
      .def("get_registered_outstanding_count", &BAM_Feature_Store<int64_t>::get_registered_outstanding_count)
      .def("get_registered_front_request_id", &BAM_Feature_Store<int64_t>::get_registered_front_request_id)
      .def("registered_front_ready", &BAM_Feature_Store<int64_t>::registered_front_ready)
      .def("get_registered_ready_front_request_id", &BAM_Feature_Store<int64_t>::get_registered_ready_front_request_id)
      .def("get_registered_front_state", &BAM_Feature_Store<int64_t>::get_registered_front_state)
      .def("get_registered_request_id_at", &BAM_Feature_Store<int64_t>::get_registered_request_id_at)
      .def("get_registered_request_state_at", &BAM_Feature_Store<int64_t>::get_registered_request_state_at)
      .def("get_registered_last_consumed_request_id", &BAM_Feature_Store<int64_t>::get_registered_last_consumed_request_id)

      .def("read_feature_merged", &BAM_Feature_Store<int64_t>::read_feature_merged)
      .def("read_feature_merged_hetero", &BAM_Feature_Store<int64_t>::read_feature_merged_hetero)

      .def("set_window_buffering", &BAM_Feature_Store<int64_t>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<int64_t>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<int64_t>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<int64_t>::flush_cache)
      .def("enable_bam_policy_cache", &BAM_Feature_Store<int64_t>::enable_bam_policy_cache)
      .def("enable_bam_policy_cache_grouped", &BAM_Feature_Store<int64_t>::enable_bam_policy_cache_grouped)
      .def("update_bam_policy_scores", &BAM_Feature_Store<int64_t>::update_bam_policy_scores)
      .def("update_bam_policy_scores_device", &BAM_Feature_Store<int64_t>::update_bam_policy_scores_device)
      .def("query_sample_residency_device", &BAM_Feature_Store<int64_t>::query_sample_residency_device)
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
