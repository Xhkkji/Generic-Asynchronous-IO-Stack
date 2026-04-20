import math
import os
import time
import queue as py_queue
import threading
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
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None, prefetch_depth=1):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.GIDS_Loader=GIDS_Loader
        
        # 自加
        self.prefetch_queue = []
        self.exhausted = False
        self.prefetch_depth = max(1, int(prefetch_depth))
        print(f"GIDS.py提交, prefetch:{self.prefetch_depth}..")
        # 先准备一个 ready 的 iter，再按预取深度继续补充 submit 队列。
        self._submit_prefetch()
        self._poll_front_prefetch()
        while len(self.prefetch_queue) < self.prefetch_depth and not self.exhausted:
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
            return False
        
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
            self.prefetch_queue.append({
                "batch": next_batch,
                "polled": False,
            })
            return True
            
        except StopIteration:
            self.exhausted = True
            return False

    def _poll_front_prefetch(self):
        if not self.prefetch_queue:
            return

        if self.prefetch_queue[0]["polled"]:
            return

        next_batch = self.prefetch_queue[0]["batch"]
        self.GIDS_Loader.read_feature_single_page_single_thread_poll(
            self.dataloader.dim, next_batch, self.GIDS_Loader.gids_device)
        self.prefetch_queue[0]["polled"] = True
    
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
        self._poll_front_prefetch()
        next_item = self.prefetch_queue.pop(0)
        next_batch = next_item["batch"]
        batch = self.GIDS_Loader.fetch_feature_get_feature_light(
            self.dataloader.dim, next_batch, self.GIDS_Loader.gids_device)
        self._poll_front_prefetch()
        while len(self.prefetch_queue) < self.prefetch_depth and not self.exhausted:
            self._submit_prefetch()
        
        return batch


class _PrefetchingIter_async_registered_try_service(object):
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None, prefetch_depth=1):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.GIDS_Loader = GIDS_Loader

        # 已经把 feature request 提交到 registered rowctx 路径的
        # 逻辑 iter 队列。
        self.prefetch_queue = []
        self.exhausted = False
        self.prefetch_depth = max(1, int(prefetch_depth))
        self.pending_batch = None
        self.pending_io_count = 0
        self.outstanding_io_count = 0
        self._returned_any_batch = False
        self.max_outstanding_ios = max(0, int(getattr(
            self.GIDS_Loader, "max_registered_outstanding_ios", 0)))
        self.try_window_size = max(1, int(getattr(
            self.GIDS_Loader, "registered_try_window_size", 1)))
        self.try_poll_fallback_loops = max(32, int(getattr(
            self.GIDS_Loader, "registered_try_fallback_loops", 256)))
        self.submit_commands_per_batch = max(1, int(getattr(
            self.GIDS_Loader, "registered_submit_commands_per_batch", 1)))
        # 控制是否启用 skip-front 预热。关闭后，只强轮询当前 front iter，
        # 后排 iter 仍然可以继续 submit，但不会被主动做 skip-front service。
        self.enable_skip_front = bool(getattr(
            self.GIDS_Loader, "registered_enable_skip_front", True))
        self.debug_registered_poll = os.environ.get(
            "GIDS_REGISTERED_DEBUG", "0") not in ("0", "", "false", "False")
        self.pending_split_item = None
        self.next_logical_iter_id = 0
        self._debug_log(
            f"init prefetch_depth={self.prefetch_depth}, "
            f"poll_mode=try_service, try_window={self.try_window_size}, "
            f"max_outstanding_ios={self.max_outstanding_ios}, "
            f"fallback_loops={self.try_poll_fallback_loops}, "
            f"submit_commands_per_batch={self.submit_commands_per_batch}, "
            f"enable_skip_front={self.enable_skip_front}"
        )

        # Warmup: 第 0 个 iter 先只保留 1 个 outstanding，避免首轮 front poll
        # 在多 outstanding 下触发底层旧问题。等第一批返回后再补满深度。
        self._submit_prefetch()
        if self.submit_commands_per_batch > 1:
            self._ensure_front_logical_item_fully_submitted()
            self._fill_prefetch_queue("warmup补提")

    def start_profiling(self, output_dir="./profiler_logs"):
        print("=== start_profiling called ===")
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.__enter__()

    def stop_profiling(self):
        print(f"=== stop_profiling called, profiler id: {id(self.profiler) if self.profiler else 'None'} ===")
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            print(self.profiler.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20
            ))

    def _debug_log(self, message):
        if self.debug_registered_poll:
            print(f"[RegisteredTryService] {message}", flush=True)

    def _next_logical_iter_id(self):
        logical_iter_id = self.next_logical_iter_id
        self.next_logical_iter_id += 1
        return logical_iter_id

    def _format_request_list(self, item):
        if "request_id" in item:
            return f"[{item['request_id']}]"
        total = len(item.get("micro_batches", []))
        request_ids = list(item.get("request_ids", []))
        request_ids.extend(["-"] * max(0, total - len(request_ids)))
        return "[" + ",".join(str(x) for x in request_ids) + "]"

    def _format_logical_item_state(self, item):
        if "request_id" in item:
            return (
                f"iterid={item.get('logical_iter_id', -1)}"
                f"[batchid=0, 已提交=1/1, 已消费=0/1, requests={self._format_request_list(item)}]"
            )
        submitted = item.get("submitted_micro_count", 0)
        total = len(item.get("micro_batches", []))
        cached = item.get("materialized_micro_count", 0)
        consumed = item.get("consumed_micro_count", 0)
        return (
            f"iterid={item.get('logical_iter_id', -1)}"
            f"[已提交batch={submitted}/{total}, 已缓存batch={cached}/{total}, 已消费batch={consumed}/{total}, "
            f"requests={self._format_request_list(item)}]"
        )

    def _debug_queue_snapshot(self, label):
        if not self.debug_registered_poll:
            return
        queue_items = [self._format_logical_item_state(item) for item in self.prefetch_queue]
        if self.pending_split_item is not None:
            queue_items.append(
                "待提交:" + self._format_logical_item_state(self.pending_split_item)
            )
        queue_state = " | ".join(queue_items) if queue_items else "empty"
        self._debug_log(f"[队列] {label} 队列={queue_state}")

    def _get_next_batch_for_submit(self):
        if self.pending_batch is not None:
            return self.pending_batch, self.pending_io_count
        try:
            # 采样和 collate 在这里完成。后面的逻辑只负责
            # feature IO、CQ service 和 ready 聚合。
            next_batch = next(self.dataloader_it)
            self.pending_batch = next_batch
            self.pending_io_count = len(next_batch[0])
            self._debug_log(f"[采样] 完成 节点数={self.pending_io_count}")
            return self.pending_batch, self.pending_io_count
        except StopIteration:
            self.exhausted = True
            self._debug_log("[采样] 结束 StopIteration")
            return None, 0

    def _can_submit_ios(self, io_count):
        if self.max_outstanding_ios <= 0:
            return True
        if self.outstanding_io_count == 0:
            return True
        return self.outstanding_io_count + io_count <= self.max_outstanding_ios

    def _split_ranges(self, total_count):
        command_count = max(1, min(self.submit_commands_per_batch, max(1, total_count)))
        base = total_count // command_count
        remainder = total_count % command_count
        ranges = []
        start = 0
        for i in range(command_count):
            size = base + (1 if i < remainder else 0)
            if size <= 0:
                continue
            end = start + size
            ranges.append((start, end))
            start = end
        return ranges

    def _slice_batch_for_micro(self, batch, start, end):
        input_nodes = batch[0][start:end]
        if type(batch) is tuple:
            return (input_nodes,)
        return [input_nodes]

    def _build_split_item(self, batch, io_count):
        micro_ranges = self._split_ranges(io_count)
        micro_batches = [
            self._slice_batch_for_micro(batch, start, end)
            for start, end in micro_ranges
        ]
        micro_io_counts = [end - start for start, end in micro_ranges]
        return {
            "logical_iter_id": self._next_logical_iter_id(),
            "batch": batch,
            "io_count": io_count,
            "micro_batches": micro_batches,
            "micro_ranges": micro_ranges,
            "micro_io_counts": micro_io_counts,
            "request_ids": [],
            "submitted_micro_count": 0,
            "consumed_micro_count": 0,
            "fully_submitted": False,
            "materialized_flags": [False] * len(micro_batches),
            "materialized_micro_count": 0,
            "materialized_return_torch": None,
        }

    def _attach_feature_to_batch(self, batch, return_torch):
        if type(batch) is tuple:
            return (*batch, return_torch)
        batch_copy = list(batch)
        batch_copy.append(return_torch)
        return batch_copy

    def _next_split_item_for_submit(self):
        if self.pending_split_item is not None:
            return self.pending_split_item
        next_batch, io_count = self._get_next_batch_for_submit()
        if next_batch is None:
            return None
        split_item = self._build_split_item(next_batch, io_count)
        self.pending_batch = None
        self.pending_io_count = 0
        self.pending_split_item = split_item
        self._debug_log(
            f"[切分] 完成 iterid={split_item['logical_iter_id']}, "
            f"batch总数={len(split_item['micro_batches'])}, "
            f"batch_ranges={split_item['micro_ranges']}"
        )
        return split_item

    def _submit_prefetch_single(self):
        if self.exhausted and self.pending_batch is None:
            self._debug_log("submit_skip exhausted=True")
            return False

        next_batch, io_count = self._get_next_batch_for_submit()
        if next_batch is None:
            return False
        if not self._can_submit_ios(io_count):
            self._debug_log(
                f"budget_block outstanding_ios={self.outstanding_io_count}, "
                f"next_ios={io_count}, budget={self.max_outstanding_ios}"
            )
            return False

        # 一个逻辑 iter 在这里对应一次 registered request 提交。
        # 返回的 request_id 后面用于跟踪 ready 状态和消费顺序。
        logical_iter_id = self._next_logical_iter_id()
        self._debug_log(
            f"[提交] 开始 iterid={logical_iter_id}, batchid=0, ios={io_count}"
        )
        request_id = self.GIDS_Loader.fetch_feature_submit_registered_rowctx(
            self.dataloader.dim,
            next_batch,
            1,
            self.GIDS_Loader.gids_device,
        )
        self.prefetch_queue.append({
            "logical_iter_id": logical_iter_id,
            "batch": next_batch,
            "request_id": request_id,
            "io_count": io_count,
        })
        self.outstanding_io_count += io_count
        self.pending_batch = None
        self.pending_io_count = 0
        self._debug_log(
            f"[提交] 完成 iterid={logical_iter_id}, batchid=0, request_id={request_id}, "
            f"ios={io_count}, "
            f"逻辑队列长度={len(self.prefetch_queue)}, "
            f"提交后outstanding={self.GIDS_Loader.get_registered_outstanding_count()}, "
            f"提交后outstanding_ios={self.outstanding_io_count}"
        )
        self._debug_queue_snapshot("提交状态")
        return True

    def _submit_prefetch_split(self):
        if self.exhausted and self.pending_batch is None and self.pending_split_item is None:
            self._debug_log("submit_skip exhausted=True")
            return False

        split_item = self._next_split_item_for_submit()
        if split_item is None:
            return False

        micro_idx = split_item["submitted_micro_count"]
        micro_io_count = split_item["micro_io_counts"][micro_idx]
        self._debug_log(
            f"[提交] 开始 iterid={split_item['logical_iter_id']}, "
            f"batchid={micro_idx}, batch_range={split_item['micro_ranges'][micro_idx]}, "
            f"ios={micro_io_count}"
        )
        if not self._can_submit_ios(micro_io_count):
            self._debug_log(
                f"budget_block outstanding_ios={self.outstanding_io_count}, "
                f"next_ios={micro_io_count}, budget={self.max_outstanding_ios}"
            )
            return False

        request_id = self.GIDS_Loader.fetch_feature_submit_registered_rowctx(
            self.dataloader.dim,
            split_item["micro_batches"][micro_idx],
            1,
            self.GIDS_Loader.gids_device,
        )

        if micro_idx == 0:
            self.prefetch_queue.append(split_item)
        split_item["request_ids"].append(request_id)
        split_item["submitted_micro_count"] += 1
        self.outstanding_io_count += micro_io_count

        if split_item["submitted_micro_count"] >= len(split_item["micro_batches"]):
            split_item["fully_submitted"] = True
            self.pending_split_item = None

        self._debug_log(
            f"[提交] 完成 iterid={split_item['logical_iter_id']}, "
            f"batchid={micro_idx}, request_id={request_id}, "
            f"batch_range={split_item['micro_ranges'][micro_idx]}, "
            f"micro_ios={micro_io_count}, logical_queue_len={len(self.prefetch_queue)}, "
            f"已提交batch={split_item['submitted_micro_count']}/{len(split_item['micro_batches'])}, "
            f"提交后outstanding={self.GIDS_Loader.get_registered_outstanding_count()}, "
            f"提交后outstanding_ios={self.outstanding_io_count}"
        )
        self._debug_queue_snapshot("提交状态")
        return True

    def _submit_prefetch(self):
        if self.submit_commands_per_batch <= 1:
            return self._submit_prefetch_single()
        return self._submit_prefetch_split()

    def _fill_prefetch_queue(self, label="fill"):
        if self.submit_commands_per_batch <= 1:
            while len(self.prefetch_queue) < self.prefetch_depth and not self.exhausted:
                if not self._submit_prefetch():
                    break
            return

        while not self.exhausted:
            if len(self.prefetch_queue) >= self.prefetch_depth and self.pending_split_item is None:
                break
            if not self._submit_prefetch():
                break
        self._debug_queue_snapshot(f"{label}")

    def _ensure_front_logical_item_fully_submitted(self):
        if self.submit_commands_per_batch <= 1 or not self.prefetch_queue:
            return
        front_item = self.prefetch_queue[0]
        while not front_item.get("fully_submitted", False):
            if not self._submit_prefetch():
                raise RuntimeError(
                    "Front logical iter is not fully submitted. Increase "
                    "GIDS_MAX_REGISTERED_OUTSTANDING_IOS or reduce "
                    "GIDS_REGISTERED_SUBMIT_COMMANDS_PER_BATCH."
                )

    def _ensure_split_return_buffer(self, item):
        if item.get("materialized_return_torch") is None:
            input_nodes = item["batch"][0]
            item["materialized_return_torch"] = torch.empty(
                [len(input_nodes), self.dataloader.dim],
                dtype=torch.float,
                device=self.GIDS_Loader.gids_device,
            ).contiguous()
        return item["materialized_return_torch"]

    def _front_pending_split_request(self):
        for item in self.prefetch_queue:
            if "micro_batches" not in item:
                continue
            micro_idx = item.get("materialized_micro_count", 0)
            if micro_idx >= item.get("submitted_micro_count", 0):
                continue
            request_id = item["request_ids"][micro_idx]
            return item, micro_idx, request_id
        return None, None, None

    def _drain_ready_front_split_prefix(self, label="预取"):
        if self.submit_commands_per_batch <= 1 or not self.prefetch_queue:
            return 0

        drained = 0
        while True:
            item, micro_idx, request_id = self._front_pending_split_request()
            if item is None:
                break
            ready_request_id = self.GIDS_Loader.get_registered_ready_front_request_id()
            if ready_request_id != request_id:
                break

            self._debug_log(
                f"[预取获取] 开始 iterid={item['logical_iter_id']}, "
                f"batchid={micro_idx}, request_id={request_id}, "
                f"batch_range={item['micro_ranges'][micro_idx]}, 来源={label}"
            )
            micro_result = self.GIDS_Loader.fetch_feature_get_feature_light_registered_rowctx(
                self.dataloader.dim,
                item["micro_batches"][micro_idx],
                self.GIDS_Loader.gids_device,
            )
            micro_ret = micro_result[-1]
            return_torch = self._ensure_split_return_buffer(item)
            start, end = item["micro_ranges"][micro_idx]
            return_torch[start:end].copy_(micro_ret)
            self.outstanding_io_count = max(
                0, self.outstanding_io_count - item["micro_io_counts"][micro_idx]
            )
            item["materialized_flags"][micro_idx] = True
            item["materialized_micro_count"] += 1
            drained += 1
            self._debug_log(
                f"[预取获取] 完成 iterid={item['logical_iter_id']}, "
                f"batchid={micro_idx}, request_id={request_id}, "
                f"已缓存batch={item['materialized_micro_count']}/{len(item['micro_batches'])}, "
                f"outstanding_ios={self.outstanding_io_count}, 来源={label}"
            )
        if drained > 0:
            self._debug_queue_snapshot(f"{label}_后缓存状态")
        return drained

    def _service_prefetch_nonblocking(self, label="service"):
        if not self.prefetch_queue:
            return 0
        if not self.enable_skip_front:
            return 0
        if self.GIDS_Loader.get_registered_outstanding_count() <= 1:
            return 0
        if self.submit_commands_per_batch <= 1 and len(self.prefetch_queue) <= 1:
            return 0

        # Skip-front service only "preheats" requests behind the current front
        # iter. The front iter itself is handled by _poll_front_prefetch().
        ready_front_before = self.GIDS_Loader.get_registered_ready_front_request_id()
        request_id = self.GIDS_Loader.service_registered_try_poll_window_skip_front(self.try_window_size)
        ready_front_after = self.GIDS_Loader.get_registered_ready_front_request_id()
        if request_id != 0 or ready_front_after != ready_front_before:
            self._debug_log(
                f"[轮询] 函数=service_registered_try_poll_window_skip_front, 事件={label}_done, "
                f"returned={request_id}, ready_front_before={ready_front_before}, "
                f"ready_front_after={ready_front_after}, "
                f"window={self.try_window_size}, outstanding={self.GIDS_Loader.get_registered_outstanding_count()}"
            )
        return request_id

    def _refresh_split_ready_state(self, label="refresh"):
        if self.submit_commands_per_batch <= 1 or not self.prefetch_queue:
            return

        outstanding_count = self.GIDS_Loader.get_registered_outstanding_count()
        request_states = {}
        for offset in range(outstanding_count):
            request_id = self.GIDS_Loader.get_registered_request_id_at(offset)
            if request_id == 0:
                continue
            request_states[request_id] = self.GIDS_Loader.get_registered_request_state_at(offset)

        ready_summary = []
        for item in self.prefetch_queue:
            if "micro_batches" not in item:
                continue
            total = len(item["micro_batches"])
            ready_flags = item.get("ready_flags")
            if ready_flags is None or len(ready_flags) != total:
                ready_flags = [False] * total
                item["ready_flags"] = ready_flags

            for micro_idx in range(total):
                if micro_idx < item.get("consumed_micro_count", 0):
                    ready_flags[micro_idx] = True
                    continue
                if micro_idx >= len(item.get("request_ids", [])):
                    ready_flags[micro_idx] = False
                    continue
                request_id = item["request_ids"][micro_idx]
                ready_flags[micro_idx] = (request_states.get(request_id, 0) == 1)

            ready_micro_count = sum(1 for flag in ready_flags if flag)
            item["ready_micro_count"] = ready_micro_count
            item["full_ready"] = (
                item.get("fully_submitted", False) and ready_micro_count == total
            )
            ready_summary.append(
                f"iterid={item.get('logical_iter_id', -1)}:"
                f"{ready_micro_count}/{total}"
            )

        if self.debug_registered_poll and ready_summary:
            self._debug_log(f"[聚合] {label} ready状态=" + " | ".join(ready_summary))

    def _service_all_submitted_and_refresh(self, label="service_all"):
        if self.submit_commands_per_batch <= 1 or not self.prefetch_queue:
            return

        outstanding_count = self.GIDS_Loader.get_registered_outstanding_count()
        if outstanding_count <= 0:
            self._refresh_split_ready_state(label)
            return

        front_request_id = self.GIDS_Loader.service_registered_try_poll()
        skip_request_id = 0
        if self.enable_skip_front and outstanding_count > 1:
            skip_request_id = self.GIDS_Loader.service_registered_try_poll_window_skip_front(outstanding_count)

        self._refresh_split_ready_state(label)
        if self.debug_registered_poll and (front_request_id != 0 or skip_request_id != 0):
            self._debug_log(
                f"[聚合] {label} front_returned={front_request_id}, "
                f"skip_returned={skip_request_id}, outstanding={self.GIDS_Loader.get_registered_outstanding_count()}"
            )

    def _wait_front_logical_item_ready(self, front_item, label="wait_front_iter"):
        if self.submit_commands_per_batch <= 1:
            return

        self._refresh_split_ready_state(f"{label}_begin")
        try_loops = 0
        while not front_item.get("full_ready", False):
            outstanding_count = self.GIDS_Loader.get_registered_outstanding_count()
            if outstanding_count <= 1:
                micro_idx = front_item["consumed_micro_count"]
                expected_request_id = front_item["request_ids"][micro_idx]
                self._debug_log(
                    f"[聚合] {label} fallback_single iterid={front_item['logical_iter_id']}, "
                    f"batchid={micro_idx}, request_id={expected_request_id}"
                )
                self.GIDS_Loader.service_registered_poll_compatible()
                self._refresh_split_ready_state(f"{label}_fallback")
                try_loops += 1
                continue

            self._service_all_submitted_and_refresh(f"{label}_loop")
            try_loops += 1
            if front_item.get("full_ready", False):
                break
            if try_loops % 64 == 0:
                self._debug_log(
                    f"[聚合] {label} retry iterid={front_item['logical_iter_id']}, "
                    f"loops={try_loops}, ready={front_item.get('ready_micro_count', 0)}/"
                    f"{len(front_item.get('micro_batches', []))}, "
                    f"outstanding={self.GIDS_Loader.get_registered_outstanding_count()}"
                )

    def _poll_front_request(self, expected_request_id, label="poll", iterid=None, batchid=None):
        ready_request_id = self.GIDS_Loader.get_registered_ready_front_request_id()
        if ready_request_id == expected_request_id:
            self._debug_log(
                f"[轮询] 函数=skip_ready, 事件={label}_skip_ready, "
                f"iterid={iterid}, batchid={batchid}, request_id={expected_request_id}"
            )
            return

        self._debug_log(
            f"[轮询] 函数=front_poll, 事件={label}_begin, "
            f"iterid={iterid}, batchid={batchid}, expected={expected_request_id}, "
            f"front={self.GIDS_Loader.get_registered_front_request_id()}, "
            f"state={self.GIDS_Loader.get_registered_front_state()}, "
            f"queue_len={len(self.prefetch_queue)}"
        )
        outstanding_count = self.GIDS_Loader.get_registered_outstanding_count()
        if outstanding_count <= 1:
            # outstanding 太浅时，退回兼容的 blocking poll。
            # 这里只是兜底路径，不是希望长期依赖的主路径。
            self._debug_log(
                f"[轮询] 函数=service_registered_poll_compatible, 事件={label}_fallback_single, "
                f"iterid={iterid}, batchid={batchid}, expected={expected_request_id}, "
                f"outstanding={outstanding_count}"
            )
            request_id = self.GIDS_Loader.service_registered_poll_compatible()
            self._debug_log(
                f"[轮询] 函数=service_registered_poll_compatible, 事件={label}_fallback_single_done, "
                f"iterid={iterid}, batchid={batchid}, returned={request_id}, "
                f"ready_front={self.GIDS_Loader.get_registered_ready_front_request_id()}, "
                f"state={self.GIDS_Loader.get_registered_front_state()}"
            )
            return
        # 主路径：优先用 CQ-driven try-service 推进 front request，
        # 并周期性预热后面的 outstanding request。
        try_loops = 0
        while True:
            request_id = self.GIDS_Loader.service_registered_try_poll()
            try_loops += 1
            ready_request_id = self.GIDS_Loader.get_registered_ready_front_request_id()
            if ready_request_id == expected_request_id:
                self._debug_log(
                    f"[轮询] 函数=service_registered_try_poll, 事件={label}_done, "
                    f"iterid={iterid}, batchid={batchid}, returned={request_id}, loops={try_loops}, "
                    f"ready_front={ready_request_id}, "
                    f"state={self.GIDS_Loader.get_registered_front_state()}"
                )
                return
            if try_loops % 8 == 0:
                if try_loops % 64 == 0:
                    self._debug_log(
                        f"[轮询] 函数=service_registered_try_poll, 事件={label}_retry, "
                        f"iterid={iterid}, batchid={batchid}, loops={try_loops}, returned={request_id}, "
                        f"ready_front={ready_request_id}, "
                        f"state={self.GIDS_Loader.get_registered_front_state()}"
                    )
                self._service_prefetch_nonblocking(f"{label}_window")
            if try_loops % self.try_poll_fallback_loops == 0:
                self._debug_log(
                    f"[轮询] 函数=service_registered_try_poll, 事件={label}_progress, "
                    f"iterid={iterid}, batchid={batchid}, loops={try_loops}, expected={expected_request_id}, "
                    f"front={self.GIDS_Loader.get_registered_front_request_id()}, "
                    f"state={self.GIDS_Loader.get_registered_front_state()}"
                )

    def _poll_front_prefetch(self, label="poll"):
        if not self.prefetch_queue:
            self._debug_log(f"{label}_skip_empty")
            return
        expected_request_id = self.prefetch_queue[0]["request_id"]
        self._poll_front_request(expected_request_id, label)

    def _consume_front_single(self):
        next_item = self.prefetch_queue.pop(0)
        next_batch = next_item["batch"]
        self._debug_log(
            f"[获取] 开始 iterid={next_item.get('logical_iter_id', -1)}, "
            f"batchid=0, request_id={next_item['request_id']}"
        )
        batch = self.GIDS_Loader.fetch_feature_get_feature_light_registered_rowctx(
            self.dataloader.dim, next_batch, self.GIDS_Loader.gids_device)
        self.outstanding_io_count = max(
            0, self.outstanding_io_count - next_item.get("io_count", len(next_batch[0])))
        self._debug_queue_snapshot("获取状态")
        return next_item, batch

    def _consume_front_split(self):
        self._ensure_front_logical_item_fully_submitted()
        self._fill_prefetch_queue("front消费前补提")
        self._drain_ready_front_split_prefix("front消费前预取")
        front_item = self.prefetch_queue[0]
        full_batch = front_item["batch"]
        return_torch = self._ensure_split_return_buffer(front_item)

        while front_item["consumed_micro_count"] < len(front_item["micro_batches"]):
            micro_idx = front_item["consumed_micro_count"]
            if front_item["materialized_flags"][micro_idx]:
                self._debug_log(
                    f"[获取] 复用缓存 iterid={front_item['logical_iter_id']}, "
                    f"batchid={micro_idx}, request_id={front_item['request_ids'][micro_idx]}, "
                    f"batch_range={front_item['micro_ranges'][micro_idx]}"
                )
                front_item["consumed_micro_count"] += 1
            else:
                expected_request_id = front_item["request_ids"][micro_idx]
                self._poll_front_request(
                    expected_request_id,
                    f"next_poll_micro{micro_idx}",
                    iterid=front_item["logical_iter_id"],
                    batchid=micro_idx,
                )
                self._debug_log(
                    f"[获取] 开始 iterid={front_item['logical_iter_id']}, "
                    f"batchid={micro_idx}, request_id={expected_request_id}, "
                    f"batch_range={front_item['micro_ranges'][micro_idx]}"
                )
                micro_batch = front_item["micro_batches"][micro_idx]
                micro_result = self.GIDS_Loader.fetch_feature_get_feature_light_registered_rowctx(
                    self.dataloader.dim, micro_batch, self.GIDS_Loader.gids_device)
                micro_ret = micro_result[-1]
                start, end = front_item["micro_ranges"][micro_idx]
                return_torch[start:end].copy_(micro_ret)
                self.outstanding_io_count = max(
                    0, self.outstanding_io_count - front_item["micro_io_counts"][micro_idx]
                )
                front_item["materialized_flags"][micro_idx] = True
                front_item["materialized_micro_count"] = max(
                    front_item["materialized_micro_count"], micro_idx + 1
                )
                front_item["consumed_micro_count"] += 1
            if front_item["consumed_micro_count"] < len(front_item["micro_batches"]):
                self._fill_prefetch_queue(
                    f"iter{front_item['logical_iter_id']}_batch{micro_idx}_后补提"
                )
                self._drain_ready_front_split_prefix(
                    f"iter{front_item['logical_iter_id']}_batch{micro_idx}_后预取"
                )

        self.prefetch_queue.pop(0)
        self._debug_queue_snapshot("获取状态")
        return front_item, self._attach_feature_to_batch(full_batch, return_torch)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.prefetch_queue and self.exhausted:
            raise StopIteration
        self._debug_log(
            f"[next] 开始 逻辑队列长度={len(self.prefetch_queue)}, "
            f"outstanding={self.GIDS_Loader.get_registered_outstanding_count()}, "
            f"outstanding_ios={self.outstanding_io_count}"
        )

        # Return order is still iter/FIFO order. We first make sure the front
        # request is ready, then materialize its feature tensor, then refill the
        # queue so later iters can overlap in the background.
        if self.submit_commands_per_batch <= 1:
            self._poll_front_prefetch("next_poll")
            self._service_prefetch_nonblocking("before_front_poll")
            next_item, batch = self._consume_front_single()
        else:
            self._fill_prefetch_queue("next开始补提")
            self._drain_ready_front_split_prefix("next开始预取")
            next_item, batch = self._consume_front_split()

        if not self._returned_any_batch:
            warmup_request_id = next_item.get("request_id", 0)
            if warmup_request_id == 0 and next_item.get("request_ids"):
                warmup_request_id = next_item["request_ids"][0]
            self._debug_log(
                f"[next] warmup完成 request_id={warmup_request_id}, "
                f"queue_len={len(self.prefetch_queue)}, "
                f"outstanding={self.GIDS_Loader.get_registered_outstanding_count()}"
            )

        self._fill_prefetch_queue("next返回后补提")
        self._drain_ready_front_split_prefix("next返回后预取")
        self._debug_log(
            f"[next] 返回 iterid={next_item.get('logical_iter_id', -1)}, "
            f"逻辑队列长度={len(self.prefetch_queue)}, "
            f"outstanding={self.GIDS_Loader.get_registered_outstanding_count()}, "
            f"outstanding_ios={self.outstanding_io_count}"
        )
        self._returned_any_batch = True
        return batch


class _PrefetchingIter_async_sample_io_pipeline(object):
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.GIDS_Loader = GIDS_Loader

        sampled_qsize = max(1, int(getattr(self.GIDS_Loader, "sample_io_sampled_queue_size", 1)))
        self.sampled_queue = py_queue.Queue(maxsize=sampled_qsize)
        self._sample_sentinel = object()
        self._worker_exc = None
        self.prefetch_queue = []
        self.exhausted = False
        self.current_batch = None

        self._sampler_thread = threading.Thread(
            target=self._sampler_loop,
            name="gids-sample-worker",
            daemon=True,
        )
        self._sampler_thread.start()

        # 先同步做出第一批 ready 数据，之后只保持一个“已提交、待轮询”的下一批。
        self._prepare_first_batch()

    def start_profiling(self, output_dir="./profiler_logs"):
        print("=== start_profiling called ===")
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.__enter__()

    def stop_profiling(self):
        print(f"=== stop_profiling called, profiler id: {id(self.profiler) if self.profiler else 'None'} ===")
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            print(self.profiler.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20
            ))

    def _sampler_loop(self):
        try:
            while True:
                batch = next(self.dataloader_it)
                self.sampled_queue.put(batch)
        except StopIteration:
            self.sampled_queue.put(self._sample_sentinel)
        except Exception as exc:
            self._worker_exc = exc
            self.sampled_queue.put(self._sample_sentinel)

    def _get_next_sampled_batch(self):
        batch = self.sampled_queue.get()
        if batch is self._sample_sentinel:
            self.exhausted = True
            if self._worker_exc is not None:
                raise self._worker_exc
            return None
        return batch

    def _try_get_next_sampled_batch_nowait(self):
        try:
            batch = self.sampled_queue.get_nowait()
        except py_queue.Empty:
            return None
        if batch is self._sample_sentinel:
            self.exhausted = True
            if self._worker_exc is not None:
                raise self._worker_exc
            return None
        return batch

    def _submit_batch(self, next_batch):
        self.GIDS_Loader.fetch_feature_submit(
            self.dataloader.dim,
            next_batch,
            1,
            self.GIDS_Loader.gids_device,
        )
        self.prefetch_queue.append({
            "batch": next_batch,
            "polled": False,
        })

    def _submit_prefetch_blocking(self):
        if self.exhausted or self.prefetch_queue:
            return False
        next_batch = self._get_next_sampled_batch()
        if next_batch is None:
            return False
        self._submit_batch(next_batch)
        return True

    def _submit_prefetch_nonblocking(self):
        if self.exhausted or self.prefetch_queue:
            return False
        next_batch = self._try_get_next_sampled_batch_nowait()
        if next_batch is None:
            return False
        self._submit_batch(next_batch)
        return True

    def _poll_front_prefetch(self):
        if not self.prefetch_queue:
            return

        if self.prefetch_queue[0]["polled"]:
            return

        next_batch = self.prefetch_queue[0]["batch"]
        self.GIDS_Loader.read_feature_single_page_single_thread_poll(
            self.dataloader.dim,
            next_batch,
            self.GIDS_Loader.gids_device,
        )
        self.prefetch_queue[0]["polled"] = True

    def _consume_front_prefetch(self):
        if not self.prefetch_queue:
            return None
        self._poll_front_prefetch()
        next_item = self.prefetch_queue.pop(0)
        next_batch = next_item["batch"]
        return self.GIDS_Loader.fetch_feature_get_feature_light(
            self.dataloader.dim,
            next_batch,
            self.GIDS_Loader.gids_device,
        )

    def _ensure_current_batch_ready(self):
        if self.current_batch is not None:
            return True

        if not self.prefetch_queue:
            if not self._submit_prefetch_blocking():
                return False

        self.current_batch = self._consume_front_prefetch()
        return self.current_batch is not None

    def _prepare_first_batch(self):
        if not self._submit_prefetch_blocking():
            self.current_batch = None
            return
        self.current_batch = self._consume_front_prefetch()
        self._submit_prefetch_nonblocking()

    def __iter__(self):
        return self

    def __next__(self):
        if not self._ensure_current_batch_ready():
            raise StopIteration

        batch = self.current_batch
        self.current_batch = None
        self._submit_prefetch_nonblocking()
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
        force_sync_read = os.environ.get("GIDS_FORCE_SYNC_READ", "0") not in ("0", "", "false", "False")
        if getattr(self.GIDS_Loader, "use_registered_try_service", False):
            return _PrefetchingIter_async_registered_try_service(
                self, super().__iter__(), GIDS_Loader=self.GIDS_Loader,
                prefetch_depth=getattr(self.GIDS_Loader, "iter_prefetch_depth", 1))
        if getattr(self.GIDS_Loader, "use_async_sample_io_pipeline", False):
            return _PrefetchingIter_async_sample_io_pipeline(
                self, super().__iter__(), GIDS_Loader=self.GIDS_Loader)
        if force_sync_read:
            return _PrefetchingIter(
                self, super().__iter__(), GIDS_Loader=self.GIDS_Loader)
        return _PrefetchingIter_async(
            self, super().__iter__(), GIDS_Loader=self.GIDS_Loader,
            prefetch_depth=getattr(self.GIDS_Loader, "iter_prefetch_depth", 1))
 
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
        self.iter_prefetch_depth = 4
        self.use_registered_try_service = os.environ.get(
            "GIDS_USE_REGISTERED_TRY_SERVICE", "0") not in ("0", "", "false", "False")
        self.registered_try_window_size = int(os.environ.get(
            "GIDS_REGISTERED_TRY_WINDOW_SIZE", "2"))
        self.registered_enable_skip_front = os.environ.get(
            "GIDS_REGISTERED_ENABLE_SKIP_FRONT", "1") not in ("0", "", "false", "False")
        self.registered_submit_commands_per_batch = int(os.environ.get(
            "GIDS_REGISTERED_SUBMIT_COMMANDS_PER_BATCH", "1"))
        self.max_registered_outstanding_ios = int(os.environ.get(
            "GIDS_MAX_REGISTERED_OUTSTANDING_IOS", "0"))
        self.use_async_sample_io_pipeline = os.environ.get(
            "GIDS_USE_ASYNC_SAMPLE_IO_PIPELINE", "0") not in ("0", "", "false", "False")
        self.sample_io_sampled_queue_size = 1
        self.sample_io_ready_queue_size = 1
        
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

    def fetch_feature_submit_registered(self, dim, batch_info, pre_fetch_num, device):
        # Compatibility alias: registered submit is now unified on the rowctx path.
        GIDS_time_start = time.time()

        index = batch_info[0].to(self.gids_device)
        index_size = len(index)
        index_ptr = index.data_ptr()
        request_id = self.BAM_FS.read_feature_submit_async_registered_rowctx(
            index_ptr, index_size, dim, self.cache_dim, 0)
        self.GIDS_submit_time += time.time() - GIDS_time_start
        return request_id

    def fetch_feature_submit_registered_rowctx(self, dim, batch_info, pre_fetch_num, device):
        GIDS_time_start = time.time()

        index = batch_info[0].to(self.gids_device)
        index_size = len(index)
        index_ptr = index.data_ptr()
        request_id = self.BAM_FS.read_feature_submit_async_registered_rowctx(
            index_ptr, index_size, dim, self.cache_dim, 0)
        self.GIDS_submit_time += time.time() - GIDS_time_start
        return request_id
    
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
        self.BAM_FS.read_feature_wait_async(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
        self.GIDS_wait_time += time.time() - GIDS_time_start

        if type(batch) is tuple:
            batch2 = (*batch, return_torch)
            return batch2
        else:
            batch.append(return_torch)
            return batch
    
    
    def read_feature_single_page_single_thread_poll(self, dim, batch, device):
        GIDS_time_start = time.time()
            
        #print("Sample  done")
        
        #print("batch 0: ", batch.ndata['_ID'])
        index = batch[0].to(self.gids_device)
        # print(f"GIDS get {index.shape[0]} nodes' features, len:{len(index)}")
        
        index_size = len(index)
        #print(batch[0])
        index_ptr = index.data_ptr()
        # return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
        self.BAM_FS.read_feature_single_page_single_thread_poll(index_ptr, index_size, dim, self.cache_dim, 0)
        self.GIDS_wait_time += time.time() - GIDS_time_start

        # if type(batch) is tuple:
        #     batch2 = (*batch, return_torch)
        #     return batch2
        # else:
        #     batch.append(return_torch)
        #     return batch

    def service_registered_poll(self):
        # Compatibility alias kept for older call sites.
        GIDS_time_start = time.time()
        request_id = self.BAM_FS.service_registered_poll_compatible()
        self.GIDS_wait_time += time.time() - GIDS_time_start
        return request_id

    def service_registered_poll_compatible(self):
        GIDS_time_start = time.time()
        request_id = self.BAM_FS.service_registered_poll_compatible()
        self.GIDS_wait_time += time.time() - GIDS_time_start
        return request_id

    def service_registered_try_poll(self):
        GIDS_time_start = time.time()
        request_id = self.BAM_FS.service_registered_try_poll()
        self.GIDS_wait_time += time.time() - GIDS_time_start
        return request_id

    def service_registered_try_poll_window_skip_front(self, window_size):
        GIDS_time_start = time.time()
        request_id = self.BAM_FS.service_registered_try_poll_window_skip_front(window_size)
        self.GIDS_wait_time += time.time() - GIDS_time_start
        return request_id

    def get_registered_outstanding_count(self):
        return self.BAM_FS.get_registered_outstanding_count()

    def get_registered_front_request_id(self):
        return self.BAM_FS.get_registered_front_request_id()

    def registered_front_ready(self):
        return self.BAM_FS.registered_front_ready()

    def get_registered_ready_front_request_id(self):
        return self.BAM_FS.get_registered_ready_front_request_id()

    def get_registered_front_state(self):
        return self.BAM_FS.get_registered_front_state()

    def get_registered_request_id_at(self, offset):
        return self.BAM_FS.get_registered_request_id_at(offset)

    def get_registered_request_state_at(self, offset):
        return self.BAM_FS.get_registered_request_state_at(offset)

    def get_registered_last_consumed_request_id(self):
        return self.BAM_FS.get_registered_last_consumed_request_id()
    
    
    # 异步轻量化轮询获取特征（不等待特征完全返回，而是先返回一个空的Tensor占位，后续通过其他机制确保特征数据被正确填充）
    def fetch_feature_get_feature_light(self, dim, batch, device):
        GIDS_time_start = time.time()
            
        #print("Sample  done")
        
        #print("batch 0: ", batch.ndata['_ID'])
        index = batch[0].to(self.gids_device)
        # print(f"GIDS get {index.shape[0]} nodes' features, len:{len(index)}")
        
        index_size = len(index)
        #print(batch[0])
        index_ptr = index.data_ptr()
        return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
        self.BAM_FS.read_feature_get_feature_light(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
        self.GIDS_wait_time += time.time() - GIDS_time_start

        if type(batch) is tuple:
            batch2 = (*batch, return_torch)
            return batch2
        else:
            batch.append(return_torch)
            return batch

    def fetch_feature_get_feature_light_registered(self, dim, batch, device):
        # Compatibility alias: registered get is now unified on the rowctx path.
        GIDS_time_start = time.time()

        index = batch[0].to(self.gids_device)
        index_size = len(index)
        return_torch = torch.zeros([index_size, dim], dtype=torch.float, device=self.gids_device).contiguous()
        request_id = self.BAM_FS.read_feature_get_feature_light_registered_rowctx(return_torch.data_ptr())
        self.GIDS_wait_time += time.time() - GIDS_time_start

        if type(batch) is tuple:
            batch2 = (*batch, return_torch)
            return batch2
        else:
            batch.append(return_torch)
            return batch

    def fetch_feature_get_feature_light_registered_rowctx(self, dim, batch, device):
        GIDS_time_start = time.time()

        index = batch[0].to(self.gids_device)
        index_size = len(index)
        return_torch = torch.zeros([index_size, dim], dtype=torch.float, device=self.gids_device).contiguous()
        request_id = self.BAM_FS.read_feature_get_feature_light_registered_rowctx(return_torch.data_ptr())
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
