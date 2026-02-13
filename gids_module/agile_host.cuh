// #include <iostream>
// #include <string>
// #include <vector>
// #include <fstream>
// #include <thread>
// #include <atomic>
// #include <functional>
// #include <chrono>

// #include <cstdio>
// #include <bam_nvme.h>

// class AgileHost {
//     std::thread monitorThread;
//     std::atomic<bool> stopFlag;

//     // unsigned int finished_polling;
//     unsigned int *stop_signal;

//     unsigned int compute_blocks;
//     unsigned int threads_per_block;
//     unsigned int agile_blocks;

//     unsigned int gpu_device_idx;

//     void * h_gpu_ptr;

//     unsigned int run;
//     unsigned int total_pairs;
//     cudaStream_t agile_cq;

// public:
// #if ENABLE_LOGGING
//     AgileLogger *h_logger;
// #endif
//     unsigned int block_size;

//     __host__ AgileHost(unsigned int gpu_device_idx, unsigned int block_size) : block_size(block_size), gpu_device_idx(gpu_device_idx) {
        
//         // cuda_err_chk(cudaMalloc(&(agile_stop_signal), sizeof(unsigned int)));
//         this->run = 0;
//         this->total_pairs = 0;
        
//     }

//     __host__ void startAgile(){
//         // unsigned int stop = 0;
//         // cuda_err_chk(cudaMemcpy(agile_stop_signal, &stop, sizeof(unsigned int), cudaMemcpyHostToDevice));
//         cuda_err_chk(cudaStreamCreateWithFlags(&(this->agile_cq), cudaStreamNonBlocking));
//         *((volatile unsigned int*)this->stop_signal) = 0;
//         unsigned int warps = this->total_pairs;
//         unsigned int threads = warps * 32;
//         unsigned int blocks = threads / 1024 + (threads % 1024 == 0 ? 0 : 1);
//         std::cout << "agile blocks: " << blocks << " threads: " << threads << std::endl;
//         start_agile_cq_service<<<blocks, min(threads, 1024), 0, this->agile_cq>>>();
//         usleep(100);
//     }
    
//     __host__ void startAgile(unsigned int griddim, unsigned int blockdim){
//         cuda_err_chk(cudaStreamCreateWithFlags(&(this->agile_cq), cudaStreamNonBlocking));
//         *((volatile unsigned int*)this->stop_signal) = 0;
//         std::cout << "agile blocks: " << griddim << " threads: " << blockdim << std::endl;
//         start_agile_cq_service<<<griddim, blockdim, 0, this->agile_cq>>>();
//         usleep(100);

//     }

//     __device__ void start_agile_cq_service()
//     {
//         unsigned int stop_sig = 0;
//         unsigned int warp_id = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
//         unsigned int warp_idx = threadIdx.x % 32;
//         unsigned int queue_idx = warp_id;
//         do {
//             unsigned int mask = __ballot_sync(0xFFFFFFFF, 1);
//             bool valid = this->warpService2(warp_id, warp_idx, this->list->pairs[queue_idx].cq.pos_offset, this->list->pairs[queue_idx].cq.mask); // change queue idx in the future
//             if(warp_idx == 0){
//                 stop_sig = *((volatile unsigned int *)this->stop_signal);
//             }
//             stop_sig = __shfl_sync(0xFFFFFFFF, stop_sig, 0);
//         } while (stop_sig == 0);
//     }

//     // each warp is responsible for one cq queue, check 32 cq entries in one iteration
//     __device__ bool warpService2(unsigned int queue_idx, unsigned int warp_idx, unsigned int & offset, unsigned int & mask){
//         unsigned int processed = waitCpl2(queue_idx, warp_idx, offset, mask);
//         mask = __ballot_sync(0xFFFFFFFF, processed);
//         if (mask == 0xFFFFFFFF) {
//             mask = 0;
//             offset += 32; // % depth
//             if(offset == this->list->pairs[queue_idx].cq.depth){
//                 offset = 0;
//                 this->list->pairs[queue_idx].cq.phase = (~(this->list->pairs[queue_idx].cq.phase)) & 0x1;
//             }
//             if(warp_idx == 0){
//                 atomicAdd(&(this->list->pairs[queue_idx].sq.prev_sq_pos), 32);
//                 *(this->list->pairs[queue_idx].cq.cqdb) = offset;
//             }
            
//         }
//     }


//     __host__ void stopAgile()
//     {
//         *((volatile unsigned int*)this->h_ctrl->stop_signal) = 1;
//         cuda_err_chk(cudaStreamSynchronize(this->agile_cq));
//         cuda_err_chk(cudaStreamDestroy(this->agile_cq));
// #if ENABLE_LOGGING
//         this->monitoring();
// #endif
//     }
// }