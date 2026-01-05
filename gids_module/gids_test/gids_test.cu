#include <pybind11/pybind11.h>

#include <cstdint>
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
// #include <pybind11/stl.h>
#include "gids_kernel.cu"
// #include <bafs_ptr.h>

// test_simple.cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>

// 1. 声明kernel函数
template <typename T>
void read_feature_kernel(T *array_ptr, T *tensor_ptr, int64_t *index_ptr,
                         int dim, int64_t num_idx, int cache_dim, uint64_t key_off);

// 2. 创建测试数据的主函数
int main()
{
  std::cout << "开始测试BAM Feature Store..." << std::endl;

  // 设置测试参数
  int num_iter = 2; // 测试2个请求
  int dim = 1024;   // 特征维度
  int cache_dim = 1024;
  int64_t num_indices[] = {1000, 2000}; // 每个请求的索引数量

  // 创建CUDA流
  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++)
  {
    cudaStreamCreate(&streams[i]);
  }

  // 分配设备内存
  float *d_array_ptr = nullptr;   // 你的特征数据指针
  float *d_tensor_ptr[num_iter];  // 输出张量
  int64_t *d_index_ptr[num_iter]; // 索引

  cudaMalloc(&d_array_ptr, 1000000 * dim * sizeof(float)); // 假设有100万个特征

  for (int i = 0; i < num_iter; i++)
  {
    // 分配输出张量内存
    size_t tensor_size = num_indices[i] * dim * sizeof(float);
    cudaMalloc(&d_tensor_ptr[i], tensor_size);

    // 分配索引内存
    size_t index_size = num_indices[i] * sizeof(int64_t);
    cudaMalloc(&d_index_ptr[i], index_size);

    // 生成测试索引数据 (0-999)
    std::vector<int64_t> host_indices(num_indices[i]);
    for (int64_t j = 0; j < num_indices[i]; j++)
    {
      host_indices[j] = j % 100000; // 随机索引
    }
    cudaMemcpy(d_index_ptr[i], host_indices.data(), index_size, cudaMemcpyHostToDevice);
  }

  auto start = std::chrono::high_resolution_clock::now();

  // 启动kernel
  for (int i = 0; i < num_iter; i++)
  {
    uint64_t b_size = 128; // block size
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_indices[i] + n_warp - 1) / n_warp;

    std::cout << "启动kernel " << i << ": g_size=" << g_size
              << ", b_size=" << b_size << std::endl;

    // 这里调用你的kernel
    read_feature_kernel<<<g_size, b_size, 0, streams[i]>>>(
        d_array_ptr, d_tensor_ptr[i], d_index_ptr[i],
        dim, num_indices[i], cache_dim, 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      std::cerr << "Kernel启动失败: " << cudaGetErrorString(err) << std::endl;
      return 1;
    }
  }

  // 同步所有流
  for (int i = 0; i < num_iter; i++)
  {
    cudaStreamSynchronize(streams[i]);
    std::cout << "流 " << i << " 同步完成" << std::endl;
  }

  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "总执行时间: " << duration.count() << " ms" << std::endl;

  // 清理
  for (int i = 0; i < num_iter; i++)
  {
    cudaFree(d_tensor_ptr[i]);
    cudaFree(d_index_ptr[i]);
    cudaStreamDestroy(streams[i]);
  }
  cudaFree(d_array_ptr);

  std::cout << "测试完成!" << std::endl;
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