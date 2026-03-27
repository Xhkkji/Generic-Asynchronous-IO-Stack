import math
import time
import torch
import numpy as np
import ctypes
import nvtx 

import BAM_Feature_Store

import dgl
from torch.utils.data import DataLoader
from collections.abc import Mapping

from dgl.dataloading import create_tensorized_dataset, WorkerInitWrapper, remove_parent_storage_columns
from dgl.utils import (
    recursive_apply, ExceptionWrapper, recursive_apply_pair, set_num_threads, get_num_threads,
    get_numa_nodes_cores, context_of, dtype_of)

from dgl import DGLHeteroGraph
from dgl.frame import LazyFeature
from dgl.storages import wrap_storage
from dgl.dataloading.base import BlockSampler, as_edge_prediction_sampler
from dgl import backend as F
from dgl.distributed import DistGraph
from dgl.multiprocessing import call_once_and_share

def _get_device(device):
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device

class CollateWrapper(object):
    def __init__(self, sample_func, g,  device):
        self.sample_func = sample_func
        self.g = g
        self.device = device

    def __call__(self, items):
        graph_device = getattr(self.g, 'device', None)   
        items = recursive_apply(items, lambda x: x.to(self.device))
        batch = self.sample_func(self.g, items)
        return recursive_apply(batch, remove_parent_storage_columns, self.g)


import torch.profiler
class _PrefetchingIter_async(object):
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.GIDS_Loader=GIDS_Loader
        
        # 自加
        self.prefetch_queue = []
        self.exhausted = False
        # self.prefetch_depth = 3  # 需要调大cache的大小，保证IO请求能装得下
        # for _ in range(2):
        #     print(f"执行_submit_prefetch")
        #     self._submit_prefetch()
        print("GIDS.py提交..")
        self._submit_prefetch()
        
    # 用于记录性能数据的函数
    def start_profiling(self, output_dir="./profiler_logs"):
        """启动 profiler"""
        print("=== start_profiling called ===")  # 添加
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # schedule=torch.profiler.schedule(
            #     wait=1,      # 预热1步
            #     warmup=1,    # 预热1步
            #     active=3,    # 记录3步
            #     repeat=1     # 重复1次
            # ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.__enter__()
    
    def stop_profiling(self):
        """停止 profiler"""
        print(f"=== stop_profiling called, profiler id: {id(self.profiler) if self.profiler else 'None'} ===")
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            # 打印统计
            print(self.profiler.key_averages().table(
                sort_by="cuda_time_total", 
                row_limit=20
            ))
            
    def _submit_prefetch(self):
        if self.exhausted:
            return
        
        try:
            # 从原始迭代器获取下一个batch信息（并移动指针）
            next_batch = next(self.dataloader_it)
            
            # 提交异步读取
            self.GIDS_Loader.fetch_feature_submit(
                self.dataloader.dim, 
                next_batch,  # 传入具体的batch信息，不是迭代器
                1, 
                self.GIDS_Loader.gids_device
            )
            # print('submit: ', next_batch[0][:10])  # 打印提交的batch信息（如节点ID列表的前10个）
            self.prefetch_queue.append(next_batch)
            
        except StopIteration:
            self.exhausted = True
    
    def __iter__(self):
        return self

    def __next__(self):
        # 获取迭代器
        # cur_it = self.dataloader_it
        # batch = self.GIDS_Loader.fetch_feature(self.dataloader.dim, cur_it, self.GIDS_Loader.gids_device)
        # 自加
        if not self.prefetch_queue and self.exhausted:
            raise StopIteration
        
        # 获取最早提交的任务
        # print(f"Prefetch queue length: {len(self.prefetch_queue)}")  # 添加
        next_batch = self.prefetch_queue.pop(0)
        batch = self.GIDS_Loader.fetch_feature_wait(self.dataloader.dim, next_batch, self.GIDS_Loader.gids_device)
        # self.GIDS_Loader.fetch_feature_submit(self.dataloader.dim, cur_it, 1, self.GIDS_Loader.gids_device)
        self._submit_prefetch()  # 提交下一个预取任务
        
        return batch

class _PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.GIDS_Loader=GIDS_Loader
        
    # 用于记录性能数据的函数
    def start_profiling(self, output_dir="./profiler_logs"):
        """启动 profiler"""
        print("=== start_profiling called ===")  # 添加
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # schedule=torch.profiler.schedule(
            #     wait=1,      # 预热1步
            #     warmup=1,    # 预热1步
            #     active=3,    # 记录3步
            #     repeat=1     # 重复1次
            # ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.__enter__()
    
    def stop_profiling(self):
        """停止 profiler"""
        print(f"=== stop_profiling called, profiler id: {id(self.profiler) if self.profiler else 'None'} ===")
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            # 打印统计
            print(self.profiler.key_averages().table(
                sort_by="cuda_time_total", 
                row_limit=20
            ))
            
    
    def __iter__(self):
        return self

    def __next__(self):
        # 获取迭代器
        cur_it = self.dataloader_it
        batch = self.GIDS_Loader.fetch_feature(self.dataloader.dim, cur_it, self.GIDS_Loader.gids_device)
        
        return batch



class GIDS_DGLDataLoader(torch.utils.data.DataLoader):

    def __init__(self, graph, indices, graph_sampler, batch_size, dim, GIDS, device=None, use_ddp=False,
                 ddp_seed=0, drop_last=False, shuffle=False,
                 use_alternate_streams=None,
                 
                 **kwargs):

        use_uva = False
        self.GIDS_Loader = GIDS
        self.dim = dim


        if isinstance(kwargs.get('collate_fn', None), CollateWrapper):
            assert batch_size is None       # must be None
            # restore attributes
            self.graph = graph
            self.indices = indices
            self.graph_sampler = graph_sampler
            self.device = device
            self.use_ddp = use_ddp
            self.ddp_seed = ddp_seed
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.use_alternate_streams = use_alternate_streams
            self.use_uva = use_uva
            kwargs['batch_size'] = None
            super().__init__(**kwargs)
            return


        if isinstance(graph, DistGraph):
            raise TypeError(
                'Please use dgl.dataloading.DistNodeDataLoader or '
                'dgl.datalaoding.DistEdgeDataLoader for DistGraphs.')
  
        self.graph = graph
        self.indices = indices     
        num_workers = kwargs.get('num_workers', 0)

        indices_device = None
        try:
            if isinstance(indices, Mapping):
                indices = {k: (torch.tensor(v) if not torch.is_tensor(v) else v)
                           for k, v in indices.items()}
                indices_device = next(iter(indices.values())).device
            else:
                indices = torch.tensor(indices) if not torch.is_tensor(indices) else indices
                indices_device = indices.device
        except:     # pylint: disable=bare-except
            # ignore when it fails to convert to torch Tensors.
            pass

        if indices_device is None:
            if not hasattr(indices, 'device'):
                raise AttributeError('Custom indices dataset requires a \"device\" \
                attribute indicating where the indices is.')
            indices_device = indices.device

        if device is None:     
            device = torch.cuda.current_device()
        self.device = _get_device(device)

        # Sanity check - we only check for DGLGraphs.
        if isinstance(self.graph, DGLHeteroGraph):            
            self.graph.create_formats_()
            if not self.graph._graph.is_pinned():
                self.graph._graph.pin_memory_()
            

            # Check use_alternate_streams
            if use_alternate_streams is None:
                use_alternate_streams = (
                    self.device.type == 'cuda' and self.graph.device.type == 'cpu' and
                    not use_uva)

        if (torch.is_tensor(indices) or (
                isinstance(indices, Mapping) and
                all(torch.is_tensor(v) for v in indices.values()))):
            self.dataset = create_tensorized_dataset(
                indices, batch_size, drop_last, use_ddp, ddp_seed, shuffle,
                kwargs.get('persistent_workers', False))
        else:
            self.dataset = indices

        self.ddp_seed = ddp_seed
        self.use_ddp = use_ddp
        self.use_uva = use_uva
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.graph_sampler = graph_sampler
        self.use_alternate_streams = use_alternate_streams


        self.cpu_affinity_enabled = False

        worker_init_fn = WorkerInitWrapper(kwargs.get('worker_init_fn', None))

        self.other_storages = {}

        super().__init__(
            self.dataset,
            collate_fn=CollateWrapper(
                self.graph_sampler.sample, graph, self.device),
            batch_size=None,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            **kwargs)

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()
        # When using multiprocessing PyTorch sometimes set the number of PyTorch threads to 1
        # when spawning new Python threads.  This drastically slows down pinning features.
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        return _PrefetchingIter_async(
            self, super().__iter__(), GIDS_Loader=self.GIDS_Loader)
 
    def print_stats(self):
        self.GIDS_Loader.print_stats()

    def print_timer(self):
        #if(self.bam):
        #     print("feature aggregation time test: %f" % self.sample_time)
        #print("graph travel time: %f" % self.graph_travel_time)
        self.sample_time = 0.0
        self.graph_travel_time = 0.0

class GIDS():
    def __init__(self, page_size=4096, off=0, cache_dim = 1024, num_ele = 300*1000*1000*1024, 
        num_ssd = 1,  ssd_list = None, cache_size = 10,  
        ctrl_idx=0, 
        window_buffer=False, wb_size = 8, 
        accumulator_flag = False, 
        long_type=False, 
        heterograph=False,
        heterograph_map=None):

        #self.sample_type = "LADIES"

        if(long_type):
            self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store_long()
        else:
            self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store_float()
        
        # CPU Buffer and Storage Access Accumulator Metadata
        self.accumulator_flag = accumulator_flag
        self.required_accesses = 0
        self.prev_cpu_access = 0
        self.return_torch_buffer = []
        self.index_list = []
        

        # Window Buffering MetaData
        self.window_buffering_flag = window_buffer
        self.window_buffer = []
        self.wb_init = False
        self.wb_size = wb_size
        
        # 异步IO预提交数量
        self.pre_submit_buffer_flag = True
        self.pre_submit_buffer = []
        self.pre_submit_buffer_isInit = False
        self.pre_submit__buffer_size = 1
        self.pre_fetch_list = []
        
        # Cache Parameters
        self.page_size = page_size
        self.off = math.ceil(math.ceil(off / page_size)/num_ssd)
        self.num_ele = num_ele
        self.cache_size = cache_size
       
        #True if the graph is heterogenous graph
        self.heterograph = heterograph
        self.heterograph_map = heterograph_map
        self.graph_GIDS = None

        self.cache_dim = cache_dim
        self.gids_device="cuda:" + str(ctrl_idx)

        
        self.GIDS_controller = BAM_Feature_Store.GIDS_Controllers()

        if (ssd_list == None):
            print("SSD are not assigned")
            self.ssd_list = [i for i in range(num_ssd)] 
        else:
            self.ssd_list = ssd_list

        print("ssd list: ", ssd_list)
        self.GIDS_controller.init_GIDS_controllers(num_ssd, 4096, 128, self.ssd_list)
        self.BAM_FS.init_controllers(self.GIDS_controller, page_size, self.off, cache_size,num_ele, num_ssd)
        
        self.GIDS_time = 0.0
        self.WB_time = 0.0
        self.GIDS_submit_time = 0.0
        self.GIDS_wait_time = 0.0




    # For Sampling GIDS operation
    def init_graph_GIDS(self, page_size, off, cache_size, num_ele, num_ssd):
        self.graph_GIDS = BAM_Feature_Store.BAM_Feature_Store_long()
        self.graph_GIDS.init_controllers(self.GIDS_controller,page_size, off, cache_size, num_ele, num_ssd)

    def get_offset_array(self):
        ret = self.graph_GIDS.get_offset_array()
        return ret

    def get_array_ptr(self):
        return self.graph_GIDS.get_array_ptr()

    # For static CPU feature buffer
    def cpu_backing_buffer(self, dim, length):
        self.BAM_FS.cpu_backing_buffer(dim, length)
        
    def set_cpu_buffer(self, ten, N):
        topk_ten = ten[:N]
        topk_len = len(topk_ten)
        d_ten = topk_ten.to(self.gids_device)
        self.BAM_FS.set_cpu_buffer(d_ten.data_ptr(), topk_len)

    # Window Buffering
    def window_buffering(self, batch):
        s_time = time.time()
        if(self.heterograph):    
             for key, value in batch[0].items():            
                if(len(value) == 0):
                    next
                else:
                    s_time = time.time()
                    input_tensor = value.to(self.gids_device)
                    key_off = 0
                    if(self.heterograph_map != None):
                        if (key in self.heterograph_map):
                            key_off = self.heterograph_map[key]
                        else:
                            print("Cannot find key: ", key, " in the heterograph map!")
                        
                    num_pages = len(input_tensor)
                    self.BAM_FS.set_window_buffering(input_tensor.data_ptr(), num_pages, key_off)
                    e_time = time.time()
                    self.WB_time += e_time - s_time
        
        else:
            input_tensor = batch[0].to(self.gids_device)
            num_pages = len(input_tensor)
            self.BAM_FS.set_window_buffering(input_tensor.data_ptr(), num_pages, 0)
            e_time = time.time()
            self.WB_time += e_time - s_time
            

    # Window Buffering Helper Function    
    def fill_wb(self, it, num):
        for i in range(num):
            batch = next(it)
            self.window_buffer.append(batch)
            #run window buffering for the current batch
            self.window_buffering(batch)
    
    def fill_pre_submit_buffer(self, it, num):
        for i in range(num):
            batch = next(it)
            self.pre_submit_buffer.append(batch)
        

    # BW in GB/s, latency in micro seconds
    def set_required_storage_access(self, bw, l_ssd, l_system, num_ssd, p):
        accesses = (p * bw * 1024 / self.page_size * (l_ssd + l_system) * num_ssd) / (1-p)
        self.required_accesses = accesses
        print("Number of required storage accesses: ", accesses)

    # 异步提交(仅提交一个请求，多请求在prefetch函数中实现)
    def fetch_feature_submit(self, dim, batch_info, pre_fetch_num, device):
        GIDS_time_start = time.time()
        
        # if self.pre_submit_buffer_flag:
        #     #Filling up the pre submit buffer
        #     if(self.pre_submit_buffer_isInit == False):
        #         self.fill_pre_submit_buffer(it, self.pre_submit__buffer_size)
        #         self.pre_submit_buffer_isInit = True
            
        #print("Sample  start")
        # batch_info 是具体的batch信息（如节点ID列表）
        
        #print("Sample  done")
        # batch = self.pre_fetch_list[i]
        #print("batch 0: ", batch.ndata['_ID'])
        
        index = batch_info[0].to(self.gids_device)
        index_size = len(index)
        #print(batch[0])
        index_ptr = index.data_ptr()
        # self.BAM_FS.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
        self.BAM_FS.read_feature_submit_async(index_ptr, index_size, dim, self.cache_dim, 0)
        # self.BAM_FS.read_feature_wait_async(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
        self.GIDS_submit_time += time.time() - GIDS_time_start
        # print(f"Pre-submitted index.len{index_size}, index:{index[:10]}...")
        return index_size
    
    # 异步获取
    def fetch_feature_wait(self, dim, batch, device):
        GIDS_time_start = time.time()
            
        #print("Sample  done")
        
        #print("batch 0: ", batch.ndata['_ID'])
        index = batch[0].to(self.gids_device)
        # print(f"GIDS get {index.shape[0]} nodes' features, len:{len(index)}")
        
        index_size = len(index)
        #print(batch[0])
        index_ptr = index.data_ptr()
        return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
        # self.BAM_FS.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
        # self.BAM_FS.read_feature_submit_async(index_ptr, index_size, dim, self.cache_dim, 0)
        self.BAM_FS.read_feature_wait_async(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
        self.GIDS_wait_time += time.time() - GIDS_time_start

        if type(batch) is tuple:
            batch2 = (*batch, return_torch)
            return batch2
        else:
            batch.append(return_torch)
            return batch

    
    #Fetching Data from the SSDs
    def fetch_feature(self, dim, it, device):
        GIDS_time_start = time.time()

        if(self.window_buffering_flag):
            #Filling up the window buffer
            if(self.wb_init == False):
                self.fill_wb(it, self.wb_size)
                self.wb_init = True

        #print("Sample  start")
        next_batch = next(it)
        #print("Sample  done")

        self.window_buffer.append(next_batch)
        #Update Counters for Windwo Buffering
        if(self.window_buffering_flag):
            self.window_buffering(next_batch)
        
        # When the Storage Access Accumulator is enabled
        if(self.accumulator_flag):
            index_size_list = []
            index_ptr_list = []
            return_torch_list = []
            key_list = []

            if(len(self.return_torch_buffer) != 0):
                return_ten = self.return_torch_buffer.pop(0)
                return_batch = self.window_buffer.pop(0)
                # 自加##############################################
                if isinstance(return_batch, tuple):
                    return_batch = list(return_batch)
                # 自加##############################################
                return_batch.append(return_ten)
                self.GIDS_time += time.time() - GIDS_time_start
                return return_batch

            buffer_size = len(self.window_buffer)
            current_access = 0
            num_iter = 0
            required_accesses = self.required_accesses


            if(self.heterograph):
                while(1):
                    if(num_iter >= buffer_size):
                        batch = next(it)
                        for k , v in batch[0].items():
                            current_access += len(v)
                        
                        self.window_buffer.append(batch)
                        if(self.window_buffering_flag):
                            self.window_buffering(batch)

                    else:
                        batch = self.window_buffer[num_iter]
                        for k , v in batch[0].items():
                            current_access += len(v)

                    num_iter +=1
                    required_accesses += self.prev_cpu_access
                    if(current_access > (required_accesses )):
                        break

                num_concurrent_iter = 0
                for i in range(num_iter):
                    batch = self.window_buffer[i]
                    ret_ten = {}
                    for k , v in batch[0].items():
                        if(len(v) == 0):
                            empty_t = torch.empty((0, dim)).to(self.gids_device)
                            ret_ten[k] = empty_t
                        else:
                            key_off = 0
                            if(self.heterograph_map != None):
                                if (k in self.heterograph_map):
                                    key_off = self.heterograph_map[k]
                                else:
                                    print("Cannot find key: ", k, " in the heterograph map!")
                            v = v.to(self.gids_device)
                            index_size = len(v)
                            index_size_list.append(index_size)
                            return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device)
                            index_ptr_list.append(v.data_ptr())
                            ret_ten[k] = return_torch
                            return_torch_list.append(return_torch.data_ptr())
                            key_list.append(key_off)
                            num_concurrent_iter += 1
                    self.return_torch_buffer.append(ret_ten)
                self.BAM_FS.read_feature_merged_hetero(num_concurrent_iter, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim, key_list)

                return_ten = self.return_torch_buffer.pop(0)
                return_b = self.window_buffer.pop(0)
                if type(return_b) is tuple:
                    return_batch = (*return_b, return_ten) 
                else:
                    return_batch = return_b
                    return_batch.append(return_ten)
                self.GIDS_time += time.time() - GIDS_time_start

                cpu_access_count = self.BAM_FS.get_cpu_access_count()
                self.prev_cpu_access = int(cpu_access_count / num_iter)
                self.BAM_FS.flush_cpu_access_count()

                return return_batch
            else:
                while(1):
                    # 若未命中就从迭代器获取，否则从窗口缓存获取
                    if(num_iter >= buffer_size):
                        batch = next(it)
                        current_access += len(batch[0])
                        self.window_buffer.append(batch)
                        if(self.window_buffering_flag):
                            self.window_buffering(batch)
                    else:
                        batch = self.window_buffer[num_iter]
                        current_access += len(batch[0])
                    num_iter +=1
                    required_accesses += self.prev_cpu_access
                    if(current_access > (required_accesses )):
                        break
                
                # num_iter为当前批次要求的数量，计算得来
                for i in range(num_iter):
                    batch = self.window_buffer[i]
                    index = batch[0].to(self.gids_device)
                    index_size = len(index)
                    index_size_list.append(index_size)
                    # 存储要求结点的特征
                    return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device)
                    index_ptr_list.append(index.data_ptr())
                    return_torch_list.append(return_torch.data_ptr())  # 大小为批次数量，存储所有批次对应的二维数组[index_num[i], feature_dim]指针
                    self.return_torch_buffer.append(return_torch)  # 存储所有批次的结点特征,[num_iter, index_num[i], feature_dim]
                # 此处合并所有批次的数据，方便一次取出(下方的pop(0))
                self.BAM_FS.read_feature_merged(num_iter, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim)
                # print(f"return_torch_list.len:{len(return_torch_list)}")
                # len(return_torch_list):1, 输出结点数量,即所有结点特征对应的数据地址
                return_ten = self.return_torch_buffer.pop(0)
                return_b = self.window_buffer.pop(0)
                if type(return_b) is tuple:
                    return_batch = (*return_b, return_ten)
                else:
                    return_batch = return_b
                    return_batch.append(return_ten)


                self.GIDS_time += time.time() - GIDS_time_start

                cpu_access_count = self.BAM_FS.get_cpu_access_count()
                self.prev_cpu_access = int(cpu_access_count / num_iter)
                self.BAM_FS.flush_cpu_access_count()
                
                # print(f"return_batch[0].shape:{return_batch[0].shape}")  # 节点索引
                # print(f"return_batch[1].shape:{return_batch[1].shape}")  
                # print(f"return_batch[2].len:{len(return_batch[2])}")  # list
                # print(f"return_batch[3].shape:{return_batch[3].shape}") # 特征
                # len(return_batch):4,元组
                # return_batch[0].shape:torch.Size([177754])
                # return_batch[1].shape:torch.Size([1024])
                # return_batch[2].len:3
                # return_batch[3].shape:torch.Size([177754, 1024])
                # 1234567
                return return_batch
        
        # Storage Access Accumulator is disabled
        else:
            if(self.heterograph):
                batch = self.window_buffer.pop(0)
                ret_ten = {}
                index_size_list = []
                index_ptr_list = []
                return_torch_list = []
                key_list = []
                
                num_keys = 0
                for key , v in batch[0].items():
                    if(len(v) == 0):
                        empty_t = torch.empty((0, dim)).to(self.gids_device).contiguous()
                        ret_ten[key] = empty_t
                    else:
                        key_off = 0
                        if(self.heterograph_map != None):
                            if (key in self.heterograph_map):
                                key_off = self.heterograph_map[key]
                            else:
                                print("Cannot find key: ", key, " in the heterograph map!")
                        
                        g_index = v.to(self.gids_device)
                        index_size = len(g_index)
                        index_ptr = g_index.data_ptr()
                        
                        return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
                        return_torch_list.append(return_torch.data_ptr())
                        ret_ten[key] = return_torch
                        num_keys += 1
                        index_ptr_list.append(index_ptr)
                        index_size_list.append(index_size)
                        key_list.append(key_off)

                self.BAM_FS.read_feature_hetero(num_keys, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim, key_list)

                self.GIDS_time += time.time() - GIDS_time_start
                if type(batch) is tuple:
                    batch2 = (*batch, ret_ten)
                    return batch2
                else:
                    batch.append(ret_ten)
                    return batch

            else:
                batch = self.window_buffer.pop(0)
                #print("batch 0: ", batch.ndata['_ID'])
                index = batch[0].to(self.gids_device)
                index_size = len(index)
                #print(batch[0])
                index_ptr = index.data_ptr()
                return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
                self.BAM_FS.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
                # self.BAM_FS.read_feature_submit_async(index_ptr, index_size, dim, self.cache_dim, 0)
                # self.BAM_FS.read_feature_wait_async(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
                self.GIDS_time += time.time() - GIDS_time_start

                if type(batch) is tuple:
                    batch2 = (*batch, return_torch)
                    return batch2
                else:
                    batch.append(return_torch)
                    return batch



    def print_stats(self):
        print("GIDS time: ", self.GIDS_time)
        print("GIDS_submit_time: ", self.GIDS_submit_time)
        print("GIDS_wait_time: ", self.GIDS_wait_time)
        wbtime = self.WB_time 
        print("WB time: ", wbtime)
        self.WB_time = 0.0
        self.GIDS_time = 0.0
        self.GIDS_submit_time = 0.0
        self.GIDS_wait_time = 0.0
        self.BAM_FS.print_stats()
        
        if (self.graph_GIDS != None):
            self.graph_GIDS.print_stats_no_ctrl()
        return

    # Utility FUnctions
    def store_tensor(self, in_ten, offset):
        num_e = len(in_ten)
        self.BAM_FS.store_tensor(in_ten.data_ptr(),num_e,offset)

    def store_mmap_tensor(self, in_ten, offset):

        #y = in_ten[:200000].copy()
        y = in_ten.copy()
        print(y)
        print(y.flags)
        # for i in range(100):
        #     print("Tensor val: ", y[i])

        num_e = len(y)
        print("num ele: ", num_e, " ptr: ", y.ctypes.data)
        self.BAM_FS.store_tensor(y.ctypes.data,num_e,offset)

    def read_tensor(self, num, offset):
        self.BAM_FS.read_tensor(num, offset)

    def flush_cache(self):
        self.BAM_FS.flush_cache()


