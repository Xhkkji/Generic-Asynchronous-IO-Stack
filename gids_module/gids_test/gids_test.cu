#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stdio.h>

#include <bam_nvme.h>
// #include <pybind11/stl.h>
#include "gids_kernel.cu"
// #include <bafs_ptr.h>

// test_simple.cpp

#include <chrono>
#include <cuda_runtime.h>

// 1. 声明要测试的kernel
template <typename T>
__global__ void read_feature_kernel_with_cpu_backing_memory(
    void *dr, void *range, T *out_tensor_ptr,
    int64_t *index_ptr, int dim,
    int64_t num_idx, int cache_dim,
    // 以下保持原参数
    struct GIDS_CPU_buffer<T> CPU_buffer,
    bool cpu_seq,
    unsigned int *d_cpu_access,
    uint64_t key_off);

// 2. 定义必要的结构体（最小化）
// template <typename T>
// struct GIDS_CPU_buffer
// {
//   T *device_cpu_buffer;
//   int cpu_buffer_len;
// };

// 3. 主测试函数
int main()
{
  std::cout << "=== 测试 read_feature_kernel_with_cpu_backing_memory ===" << std::endl;

  // 设置参数（完全保持原样）
  int dim = 1024;
  int cache_dim = 1024;
  int64_t num_idx = 100; // 测试100个索引
  bool cpu_seq = false;  // 使用位图模式

  // 分配和初始化CPU buffer（模拟）
  GIDS_CPU_buffer<float> CPU_buffer;
  CPU_buffer.cpu_buffer_len = 50000; // 假设CPU buffer有5万个特征
  cudaMalloc(&CPU_buffer.device_cpu_buffer,
             CPU_buffer.cpu_buffer_len * cache_dim * sizeof(float));
  cudaMemset(CPU_buffer.device_cpu_buffer, 0,
             CPU_buffer.cpu_buffer_len * cache_dim * sizeof(float));

  // 分配range对象内存（模拟）
  void *d_range;
  cudaMalloc(&d_range, 1024); // 分配1KB模拟range对象
  cudaMemset(d_range, 0, 1024);

  // 分配dr对象内存（模拟）
  void *dr;
  cudaMalloc(&dr, 1024);
  cudaMemset(dr, 0, 1024);

  // 分配输出张量
  float *out_tensor_ptr;
  cudaMalloc(&out_tensor_ptr, num_idx * dim * sizeof(float));

  // 分配索引数组
  int64_t *index_ptr;
  cudaMalloc(&index_ptr, num_idx * sizeof(int64_t));

  // 填充测试索引（随机0-99999）
  std::vector<int64_t> host_indices(num_idx);
  for (int i = 0; i < num_idx; i++)
  {
    host_indices[i] = rand() % 100000; // 随机索引
  }
  cudaMemcpy(index_ptr, host_indices.data(),
             num_idx * sizeof(int64_t), cudaMemcpyHostToDevice);

  // 分配cpu_access计数器
  unsigned int *d_cpu_access;
  cudaMalloc(&d_cpu_access, sizeof(unsigned int));
  cudaMemset(d_cpu_access, 0, sizeof(unsigned int));

  uint64_t key_off = 0; // 偏移为0

  // =========== 启动kernel（完全保持原样）===========
  int b_size = 128; // block大小
  int n_warp = b_size / 32;
  int g_size = (num_idx + n_warp - 1) / n_warp;

  std::cout << "启动参数: grid=" << g_size
            << ", block=" << b_size
            << ", num_idx=" << num_idx
            << ", dim=" << dim << std::endl;

  // 直接调用要测试的kernel
  read_feature_kernel_with_cpu_backing_memory<float>
      <<<g_size, b_size, 0, 0>>>(
          dr, d_range, out_tensor_ptr,
          index_ptr, dim,
          num_idx, cache_dim,
          CPU_buffer,
          cpu_seq,
          d_cpu_access,
          key_off);

  // 检查错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cout << "❌ Kernel启动失败: "
              << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  std::cout << "✅ Kernel启动成功" << std::endl;

  // 同步（保持原样）
  cudaDeviceSynchronize();
  cudaDeviceSynchronize();

  // 检查是否有运行时错误
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cout << "❌ 运行时错误: "
              << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  std::cout << "✅ 测试通过！" << std::endl;

  // 清理
  cudaFree(CPU_buffer.device_cpu_buffer);
  cudaFree(d_range);
  cudaFree(dr);
  cudaFree(out_tensor_ptr);
  cudaFree(index_ptr);
  cudaFree(d_cpu_access);

  return 0;
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

// template <typename TYPE>
// void BAM_Feature_Store<TYPE>::read_feature_merged(int num_iter, const std::vector<uint64_t> &i_ptr_list, const std::vector<uint64_t> &i_index_ptr_list,
//                                                   const std::vector<uint64_t> &num_index, int dim, int cache_dim = 1024)
// {
//   printf("num_iter:%d\n", num_iter);
//   cudaStream_t streams[num_iter];
//   for (int i = 0; i < num_iter; i++)
//   {
//     cudaStreamCreate(&streams[i]);
//   }

//   cuda_err_chk(cudaDeviceSynchronize());
//   auto t1 = Clock::now();

//   for (uint64_t i = 0; i < num_iter; i++)
//   {
//     uint64_t i_ptr = i_ptr_list[i]; // 获取当前iter的[index_num[i], feature_dim]二维数组指针
//     uint64_t i_index_ptr = i_index_ptr_list[i];
//     TYPE *tensor_ptr = (TYPE *)i_ptr; // [index_num[i], feature_dim]二维数组指针
//     int64_t *index_ptr = (int64_t *)i_index_ptr;

//     uint64_t b_size = blkSize;
//     uint64_t n_warp = b_size / 32;
//     uint64_t g_size = (num_index[i] + n_warp - 1) / n_warp;

//     printf("g_size:%d, b_size:%d", g_size, b_size);

//     if (cpu_buffer_flag == false)
//     {
//       read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, tensor_ptr,
//                                                                    index_ptr, dim, num_index[i], cache_dim, 0);
//     }
//     else
//     {
//       read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i]>>>(a->d_array_ptr, d_range, tensor_ptr,
//                                                                                      index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag,
//                                                                                      d_cpu_access, 0);
//       // printf("read_feature_complete..\n");
//     }
//     total_access += num_index[i];
//   }

//   for (int i = 0; i < num_iter; i++)
//   {
//     printf("cudaStreamSynchronize:%d\n", i); // ×
//     cudaStreamSynchronize(streams[i]);
//   }

//   // printf("cudaStreamSynchronize finished, num_iter:%d\n", num_iter); //
//   cuda_err_chk(cudaDeviceSynchronize());
//   cuda_err_chk(cudaDeviceSynchronize());
//   cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

//   auto t2 = Clock::now();
//   auto us = std::chrono::duration_cast<std::chrono::microseconds>(
//       t2 - t1); // Microsecond (as int)
//   auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
//       t2 - t1); // Microsecond (as int)
//   const float ms_fractional =
//       static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

//   // std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
//   //         << std::endl;

//   kernel_time += ms_fractional;

//   for (int i = 0; i < num_iter; i++)
//   {
//     cudaStreamDestroy(streams[i]);
//   }
//   return;
// }