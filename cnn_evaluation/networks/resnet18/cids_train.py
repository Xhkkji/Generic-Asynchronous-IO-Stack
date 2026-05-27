import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
GIDS_SETUP_ROOT = REPO_ROOT / "GIDS_Setup"
if str(GIDS_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(GIDS_SETUP_ROOT))
if str(REPO_ROOT / "cnn_evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "cnn_evaluation"))

from GIDS import CIDS, CIDSPreparedDataset, CIDS_DataLoader
from cids_resnet18 import build_resnet18
from cids_image_transforms import ImageNetBatchPreprocessor
from cache_runtime import (
    BAMPADSPlanner,
    LogicalHotsetPADSPlanner,
    ShadePADSPlanner,
    GPUBAMPolicy,
    CachedCIDSBatchLoader,
    LRUSampleCache,
    SampleImportanceTracker,
)


RESNET18_SAMPLE_PAGE_SIZE = 1 << 17


def _load_meta(prepared_root):
    # 读取 prepared dataset 的元信息，确定类别数和图片形状。
    meta_path = Path(prepared_root) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dtype_name_to_itemsize(dtype_name):
    mapping = {
        "float16": 2,
        "float32": 4,
        "uint8": 1,
        "int64": 8,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持的 prepared dtype: {dtype_name}")
    return mapping[dtype_name]


def _resnet18_sample_page_size(meta):
    # resnet18 的 BaM 粒度：
    # - 物理页固定回到 128KiB，满足当前设备的 MDTS 限制
    # - 单张 416 图像在 prepared 数据里按 524288B 对齐，对应 4 个连续 row/page
    logical_sample_bytes = int(meta["sample_dim"]) * _dtype_name_to_itemsize(meta.get("dtype", "float32"))
    storage_sample_bytes = int(meta.get("sample_bytes", logical_sample_bytes))
    if storage_sample_bytes % RESNET18_SAMPLE_PAGE_SIZE != 0:
        raise ValueError(
            f"prepared sample_bytes={storage_sample_bytes}B 不能被固定 page_size={RESNET18_SAMPLE_PAGE_SIZE}B 整除"
        )
    if storage_sample_bytes < logical_sample_bytes:
        raise ValueError(
            f"prepared sample_bytes={storage_sample_bytes}B 小于 logical sample bytes={logical_sample_bytes}B"
        )
    return RESNET18_SAMPLE_PAGE_SIZE, logical_sample_bytes


class _TorchPreparedDataset(Dataset):
    # 使用 PyTorch 原生 Dataset/DataLoader 直接从 prepared 文件读取图片。
    def __init__(self, prepared_root, torch_read_mode="mmap"):
        self.prepared_root = Path(prepared_root)
        self.meta = _load_meta(prepared_root)
        self.shape = tuple(self.meta["shape"])
        self.sample_dim = int(self.meta["sample_dim"])
        self.num_samples = int(self.meta["num_samples"])
        self.dtype_name = self.meta.get("dtype", "float32")
        self.dtype = self._numpy_dtype_from_name(self.dtype_name)
        self.itemsize = np.dtype(self.dtype).itemsize
        self.sample_bytes = int(self.meta.get("sample_bytes", self.sample_dim * self.itemsize))
        if self.sample_bytes % self.itemsize != 0:
            raise ValueError(
                f"prepared sample_bytes={self.sample_bytes} 不能整除 dtype itemsize={self.itemsize}"
            )
        self.storage_sample_dim = self.sample_bytes // self.itemsize
        if self.storage_sample_dim < self.sample_dim:
            raise ValueError(
                f"prepared storage_sample_dim={self.storage_sample_dim} 小于 sample_dim={self.sample_dim}"
            )
        self.torch_read_mode = str(torch_read_mode)
        self.labels = np.load(self.prepared_root / self.meta.get("labels_file", "labels.npy"))
        images_path = self.prepared_root / self.meta.get("images_file", "images.bin")
        if self.torch_read_mode == "mmap":
            self.images = np.memmap(
                images_path,
                dtype=self.dtype,
                mode="r",
                shape=(self.num_samples, self.storage_sample_dim),
            )
        elif self.torch_read_mode == "buffered":
            self.images = np.fromfile(images_path, dtype=self.dtype).reshape(
                self.num_samples, self.storage_sample_dim
            )
        else:
            raise ValueError(f"不支持的 torch_read_mode: {self.torch_read_mode}")

    @staticmethod
    def _numpy_dtype_from_name(dtype_name):
        mapping = {
            "float16": np.float16,
            "float32": np.float32,
            "uint8": np.uint8,
            "int64": np.int64,
        }
        if dtype_name not in mapping:
            raise ValueError(f"不支持的 prepared dtype: {dtype_name}")
        return mapping[dtype_name]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_np = np.array(self.images[idx, :self.sample_dim], copy=True)
        image = torch.from_numpy(image_np).view(*self.shape)
        if self.dtype_name == "uint8":
            image = image.to(torch.float32).div_(255.0)
        elif image.dtype != torch.float32:
            image = image.to(torch.float32)
        label = int(self.labels[idx])
        return image, label


class _PlannedIndexDataset(Dataset):
    # 用显式 sample_id 顺序重建一个轻量 index dataset。
    def __init__(self, sample_ids, labels):
        if len(sample_ids) != len(labels):
            raise ValueError("PlannedIndexDataset 需要等长的 sample_ids 和 labels")
        self.sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return int(self.sample_ids.numel())

    def __getitem__(self, idx):
        return self.sample_ids[idx], self.labels[idx]


def _build_loader(prepared_root, batch_size, shuffle, cids_loader, prefetch_depth,
                  start_sample_id=0, io_mode="registered", registered_split=1):
    # 基于 prepared dataset 构建最小 CIDS dataloader。
    dataset = CIDSPreparedDataset(prepared_root, start_sample_id=start_sample_id)
    dataloader = CIDS_DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        CIDS_Loader=cids_loader,
        prefetch_depth=prefetch_depth,
        io_mode=io_mode,
        registered_split=registered_split,
    )
    return dataset, dataloader


def _build_torch_loader(prepared_root, batch_size, shuffle, torch_read_mode="mmap"):
    # 基于 PyTorch 原生 DataLoader 构建 prepared dataset 读取路径。
    dataset = _TorchPreparedDataset(prepared_root, torch_read_mode=torch_read_mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return dataset, dataloader


def _build_index_loader(prepared_root, batch_size, shuffle, start_sample_id=0):
    # 第一阶段 sample-cache 只需要最简单的索引 batch：
    # DataLoader 返回 (sample_id, label)，真正图片仍然由 CIDS miss path 拉取。
    dataset = CIDSPreparedDataset(prepared_root, start_sample_id=start_sample_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return dataset, dataloader


def _build_planned_index_loader(epoch_pairs, batch_size):
    sample_ids = [int(sample_id) for sample_id, _ in epoch_pairs]
    labels = [int(label) for _, label in epoch_pairs]
    dataset = _PlannedIndexDataset(sample_ids, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return dataset, dataloader


def _build_planned_cids_loader(epoch_pairs, batch_size, cids_loader, prefetch_depth, io_mode, registered_split):
    # PADS 调度功能：
    # - 下一轮仍然走原有 CIDS sync/registered 读取
    # - 这里只替换 sample_id 顺序，不改底层 IO 逻辑
    sample_ids = [int(sample_id) for sample_id, _ in epoch_pairs]
    labels = [int(label) for _, label in epoch_pairs]
    dataset = _PlannedIndexDataset(sample_ids, labels)
    dataloader = CIDS_DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        CIDS_Loader=cids_loader,
        prefetch_depth=prefetch_depth,
        io_mode=io_mode,
        registered_split=registered_split,
    )
    return dataset, dataloader


def _move_labels_to_device(labels, device):
    # CIDS 当前返回的是 (images, labels) 形式，这里统一把 label 放到训练设备。
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    return labels.to(device, non_blocking=True)


def _format_cache_summary(summary):
    # 把 cache 统计整理成更容易读的一行。
    hits = int(summary.get("hits", 0))
    misses = int(summary.get("misses", 0))
    total = hits + misses
    hit_rate = (hits / total) if total > 0 else 0.0
    return (
        f"capacity={summary.get('capacity', 0)} "
        f"size={summary.get('size', 0)} "
        f"ghost_size={summary.get('ghost_size', 0)} "
        f"hits={hits} misses={misses} "
        f"inserts={summary.get('inserts', 0)} "
        f"evictions={summary.get('evictions', 0)} "
        f"skipped_admissions={summary.get('skipped_admissions', 0)} "
        f"ghost_candidates={summary.get('ghost_candidates', 0)} "
        f"ghost_admissions={summary.get('ghost_admissions', 0)} "
        f"ghost_rejections={summary.get('ghost_rejections', 0)} "
        f"hit_rate={hit_rate:.4f}"
    )


def _extract_sample_ids_from_raw_batch(raw_batch):
    # 当前第一阶段 importance 只在“原始索引 batch”上工作，
    # 即 raw_batch 里还保留了 sample_id。
    if isinstance(raw_batch, dict):
        sample_ids = raw_batch.get("sample_ids")
    elif isinstance(raw_batch, (tuple, list)) and len(raw_batch) > 0:
        sample_ids = raw_batch[0]
    else:
        sample_ids = None

    if sample_ids is None:
        return None
    if not torch.is_tensor(sample_ids):
        sample_ids = torch.as_tensor(sample_ids)
    return sample_ids.detach()


def _extract_labels_from_raw_batch(raw_batch):
    if isinstance(raw_batch, dict):
        labels = raw_batch.get("labels")
    elif isinstance(raw_batch, (tuple, list)) and len(raw_batch) > 1:
        labels = raw_batch[1]
    else:
        labels = None

    if labels is None:
        return None
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    return labels.detach()


def _extract_images_and_labels_from_raw_batch(raw_batch):
    # 训练/验证功能：
    # - 统一处理 tuple/list 和带 sample_ids 的 dict batch
    if isinstance(raw_batch, dict):
        if "images" not in raw_batch:
            raise KeyError("batch 字典必须包含 images")
        if "labels" not in raw_batch:
            raise KeyError("batch 字典必须包含 labels")
        return raw_batch["images"], raw_batch["labels"]

    if isinstance(raw_batch, (tuple, list)):
        if len(raw_batch) < 2:
            raise ValueError("batch 至少需要包含 images 和 labels")
        return raw_batch[0], raw_batch[1]

    raise ValueError("不支持的 batch 类型，无法提取 images 和 labels")


def _compute_batch_rank_scores(sample_losses):
    sample_losses = sample_losses.detach().to(torch.float32).view(-1)
    if sample_losses.numel() == 0:
        return sample_losses

    sorted_positions = torch.argsort(sample_losses, descending=False)
    rank_values = torch.log(
        torch.arange(sample_losses.numel(), device=sample_losses.device, dtype=torch.float32) + 10.0
    )
    rank_scores = torch.empty_like(sample_losses)
    rank_scores[sorted_positions] = rank_values
    return rank_scores


def _format_importance_summary(summary):
    topk_items = summary.get("topk", [])
    topk_str = ", ".join(
        f"{item['sample_id']}:{item['ema_score']:.4f}/{item['seen_count']}"
        for item in topk_items
    )
    return (
        f"tracked_samples={summary.get('tracked_samples', 0)} "
        f"avg_ema_score={summary.get('avg_ema_score', 0.0):.4f} "
        f"max_ema_score={summary.get('max_ema_score', 0.0):.4f} "
        f"topk=[{topk_str}]"
    )


def _format_bam_policy_summary(summary):
    if not summary:
        return "update_calls=0 sync_calls=0 tracked_pages=0 last_batch_pages=0"
    return (
        f"update_calls={summary.get('update_calls', 0)} "
        f"sync_calls={summary.get('sync_calls', 0)} "
        f"tracked_pages={summary.get('tracked_pages', 0)} "
        f"last_batch_pages={summary.get('last_batch_pages', 0)}"
    )


def _zero_stage_times():
    # Stage 口径：
    # - route: 策略/调度，包括预取提交、resident/hotset/shade 规划、loader 重建
    # - extract: 数据取回、4-row 组装、to(device)、预处理
    # - compute: forward/backward/update 及其结果同步/落账
    # - apply: 分数计数、importance/policy/hotset 等状态写回
    # - io_stall: 显式轮询/等待时间
    return {
        "route_time_sec": 0.0,
        "extract_time_sec": 0.0,
        "compute_time_sec": 0.0,
        "apply_time_sec": 0.0,
        "io_stall_time_sec": 0.0,
    }


def _merge_stage_times(dst, src):
    for key in dst.keys():
        dst[key] += float(src.get(key, 0.0))
    return dst


def _format_stage_times(stage_times):
    return (
        f"Route={stage_times['route_time_sec']:.4f}s "
        f"Extract={stage_times['extract_time_sec']:.4f}s "
        f"Compute={stage_times['compute_time_sec']:.4f}s "
        f"Apply={stage_times['apply_time_sec']:.4f}s "
        f"IOStall={stage_times['io_stall_time_sec']:.4f}s"
    )


def _importance_score_for_sample(importance_tracker, sample_id):
    if importance_tracker is None:
        return 0.0
    state = importance_tracker.get(sample_id)
    if state is None:
        return 0.0
    return float(state.ema_score)


def _build_sample_cache_resident_lookup(epoch_pairs, sample_cache):
    if not epoch_pairs or sample_cache is None:
        return None
    resident_lookup = {}
    for sample_id, _ in epoch_pairs:
        sample_id = int(sample_id)
        if sample_id not in resident_lookup:
            resident_lookup[sample_id] = bool(sample_cache.contains(sample_id))
    return resident_lookup


def _build_bam_resident_lookup(epoch_pairs, cids_loader):
    if not epoch_pairs or cids_loader is None:
        return None
    unique_sample_ids = list(dict.fromkeys(int(sample_id) for sample_id, _ in epoch_pairs))
    resident_mask = cids_loader.query_sample_residency(unique_sample_ids)
    return {
        sample_id: bool(is_resident)
        for sample_id, is_resident in zip(unique_sample_ids, resident_mask.view(-1).tolist())
    }


def _resident_rate(epoch_pairs, resident_lookup):
    if not epoch_pairs or not resident_lookup:
        return 0.0
    resident_count = sum(1 for sample_id, _ in epoch_pairs if resident_lookup.get(int(sample_id), False))
    return resident_count / max(1, len(epoch_pairs))


def _build_pads_next_epoch_plan(
    epoch_pairs,
    resident_lookup,
    importance_tracker,
    rep_factor=1.5,
    shuffle_seed=0,
    sampler_name="resident",
):
    if not epoch_pairs or resident_lookup is None or importance_tracker is None:
        return None, None

    resident_pairs = []
    nonresident_pairs = []
    for sample_id, label in epoch_pairs:
        pair = (int(sample_id), int(label))
        if resident_lookup.get(int(sample_id), False):
            resident_pairs.append(pair)
        else:
            nonresident_pairs.append(pair)

    if not resident_pairs or not nonresident_pairs:
        summary = {
            "base_rep_factor": float(rep_factor),
            "effective_rep_factor": float(rep_factor),
            "mode": "base",
            "resident_rate": _resident_rate(epoch_pairs, resident_lookup),
            "base_size": len(epoch_pairs),
            "resident_count": len(resident_pairs),
            "nonresident_count": len(nonresident_pairs),
            "boosted_resident": 0,
            "planned_size": len(epoch_pairs),
            "sampler": sampler_name,
        }
        return list(epoch_pairs), summary

    ranked_resident = sorted(
        resident_pairs,
        key=lambda pair: _importance_score_for_sample(importance_tracker, pair[0]),
        reverse=True,
    )
    extra_resident_budget = min(
        len(nonresident_pairs),
        max(0, int(round(len(resident_pairs) * max(0.0, float(rep_factor) - 1.0)))),
    )
    boosted_resident = ranked_resident[:extra_resident_budget]
    resident_schedule = list(ranked_resident) + list(boosted_resident)
    kept_nonresident = nonresident_pairs[: max(0, len(nonresident_pairs) - extra_resident_budget)]

    rng = random.Random(int(shuffle_seed))
    rng.shuffle(resident_schedule)
    rng.shuffle(kept_nonresident)

    planned_pairs = resident_schedule + kept_nonresident
    summary = {
        "base_rep_factor": float(rep_factor),
        "effective_rep_factor": float(rep_factor),
        "mode": "base",
        "resident_rate": len(resident_pairs) / max(1, len(epoch_pairs)),
        "base_size": len(epoch_pairs),
        "resident_count": len(resident_pairs),
        "nonresident_count": len(nonresident_pairs),
        "boosted_resident": len(boosted_resident),
        "planned_size": len(planned_pairs),
        "sampler": sampler_name,
    }
    return planned_pairs, summary


def _format_pads_summary(summary):
    return (
        f"strategy={summary.get('strategy', 'replace')} "
        f"sampler={summary.get('sampler', 'basic')} "
        f"base_rep_factor={summary.get('base_rep_factor', 0.0):.2f} "
        f"effective_rep_factor={summary.get('effective_rep_factor', 0.0):.2f} "
        f"mode={summary.get('mode', 'base')} "
        f"resident_rate={summary.get('resident_rate', 0.0):.4f} "
        f"base_size={summary.get('base_size', 0)} "
        f"resident_count={summary.get('resident_count', 0)} "
        f"nonresident_count={summary.get('nonresident_count', 0)} "
        f"resident_unique={summary.get('resident_unique', 0)} "
        f"nonresident_unique={summary.get('nonresident_unique', 0)} "
        f"boosted_resident={summary.get('boosted_resident', 0)} "
        f"resident_duplicates={summary.get('resident_duplicates', 0)} "
        f"weighted_resident_draws={summary.get('weighted_resident_draws', 0)} "
        f"replaced_nonresident={summary.get('replaced_nonresident', 0)} "
        f"batch_resident_fraction={summary.get('batch_resident_fraction', 0.0):.4f} "
        f"batch_resident_quota={summary.get('batch_resident_quota', 0)} "
        f"ghost_size={summary.get('ghost_size', 0)} "
        f"ghost_candidates={summary.get('ghost_candidates', 0)} "
        f"ghost_admissions={summary.get('ghost_admissions', 0)} "
        f"ghost_rejections={summary.get('ghost_rejections', 0)} "
        f"hotset_size={summary.get('hotset_size', 0)} "
        f"hotset_target_size={summary.get('hotset_target_size', 0)} "
        f"hotset_retained={summary.get('hotset_retained', 0)} "
        f"hotset_newcomers={summary.get('hotset_newcomers', 0)} "
        f"hotset_resident_overlap={summary.get('hotset_resident_overlap', 0)} "
        f"locality_window_batches={summary.get('locality_window_batches', 0)} "
        f"locality_hot_draws={summary.get('locality_hot_draws', 0)} "
        f"resident_hot_draws={summary.get('resident_hot_draws', 0)} "
        f"fallback_hot_draws={summary.get('fallback_hot_draws', 0)} "
        f"planned_size={summary.get('planned_size', 0)}"
    )


def _cache_hit_rate(summary):
    if not summary:
        return 0.0
    hits = int(summary.get("hits", 0))
    misses = int(summary.get("misses", 0))
    total = hits + misses
    return (hits / total) if total > 0 else 0.0


def _choose_pads_rep_factor(base_rep_factor, adaptive_enabled, observed_rate, prev_train_loss, curr_train_loss):
    base_rep_factor = max(1.0, float(base_rep_factor))
    if not adaptive_enabled:
        return base_rep_factor, "fixed"

    rep_factor = base_rep_factor
    mode = "base"

    loss_plateau = (
        prev_train_loss is not None
        and curr_train_loss >= prev_train_loss * 0.995
    )
    loss_improving = (
        prev_train_loss is not None
        and curr_train_loss <= prev_train_loss * 0.97
    )

    if observed_rate < 0.30 or loss_plateau:
        rep_factor += 0.25
        mode = "aggressive"
    elif observed_rate > 0.45 and loss_improving:
        rep_factor -= 0.25
        mode = "relaxed"

    rep_factor = min(2.0, max(1.0, rep_factor))
    return rep_factor, mode


def _start_profiler(enabled, output_dir):
    # 轻量 profiler：重点看 CPU/CUDA 时间线与执行时间，不记录 shape/memory/stack。
    if not enabled:
        return None
    os.makedirs(output_dir, exist_ok=True)
    print("=== start_profiling called ===", flush=True)
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    )
    profiler.__enter__()
    return profiler


def _stop_profiler(profiler):
    # 停止 profiler，并同时打印 CUDA 热点和 CPU 热点。
    print(
        f"=== stop_profiling called, profiler id: {id(profiler) if profiler else 'None'} ===",
        flush=True,
    )
    if profiler is None:
        return
    profiler.__exit__(None, None, None)
    try:
        key_averages = profiler.key_averages()
    except AssertionError:
        print("=== profiler has no finalized events to summarize ===", flush=True)
        return
    print("=== profiler summary: cuda_time_total ===", flush=True)
    print(
        key_averages.table(
            sort_by="cuda_time_total",
            row_limit=20,
        ),
        flush=True,
    )
    print("=== profiler summary: cpu_time_total ===", flush=True)
    print(
        key_averages.table(
            sort_by="cpu_time_total",
            row_limit=100,
        ),
        flush=True,
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    image_preprocessor,
    profiler=None,
    max_iters=None,
    cached_batch_loader=None,
    log_interval=20,
    importance_tracker=None,
    bam_policy=None,
    bam_policy_cids=None,
    global_step_offset=0,
    collect_epoch_pairs=False,
):
    # 最小训练循环：只做前向、反向、优化和简单精度统计。
    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_correct = torch.zeros((), device=device, dtype=torch.float32)
    total_samples = 0
    steps_ran = 0
    epoch_pairs = [] if collect_epoch_pairs else None
    stage_times = _zero_stage_times()
    data_iter = iter(dataloader)

    while True:
        fetch_start_time = time.perf_counter()
        try:
            raw_batch = next(data_iter)
        except StopIteration:
            break
        stage_times["extract_time_sec"] += time.perf_counter() - fetch_start_time
        step_idx = steps_ran + 1
        steps_ran = step_idx

        extract_start_time = time.perf_counter()
        sample_ids = None
        raw_labels = None
        if importance_tracker is not None or bam_policy is not None:
            sample_ids = _extract_sample_ids_from_raw_batch(raw_batch)
            if collect_epoch_pairs:
                raw_labels = _extract_labels_from_raw_batch(raw_batch)
        if cached_batch_loader is not None:
            images, labels = cached_batch_loader.fetch_batch(raw_batch)
        else:
            images, labels = _extract_images_and_labels_from_raw_batch(raw_batch)
        images = images.to(device, non_blocking=True)
        images = image_preprocessor.train(images)
        labels = _move_labels_to_device(labels, device)
        stage_times["extract_time_sec"] += time.perf_counter() - extract_start_time

        compute_start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        sample_losses = F.cross_entropy(logits, labels, reduction="none")
        loss = sample_losses.mean()
        stage_times["compute_time_sec"] += time.perf_counter() - compute_start_time

        sample_scores = None
        apply_start_time = time.perf_counter()
        if sample_ids is not None and (importance_tracker is not None or bam_policy is not None):
            sample_scores = _compute_batch_rank_scores(sample_losses)
        if importance_tracker is not None and sample_ids is not None:
            importance_tracker.update(
                sample_ids=sample_ids,
                sample_scores=sample_scores,
                step=global_step_offset + step_idx,
            )
            if cached_batch_loader is not None:
                cached_batch_loader.refresh_cached_scores(sample_ids)
            if collect_epoch_pairs and raw_labels is not None:
                epoch_pairs.extend(
                    (int(sample_id), int(label))
                    for sample_id, label in zip(
                        sample_ids.detach().cpu().view(-1).tolist(),
                        raw_labels.detach().cpu().view(-1).tolist(),
                    )
                )
        if bam_policy is not None and sample_ids is not None:
            bam_policy.update_from_batch(sample_ids, sample_scores)
            if bam_policy_cids is not None:
                bam_policy.sync_to_bam(bam_policy_cids)
        stage_times["apply_time_sec"] += time.perf_counter() - apply_start_time

        compute_suffix_start_time = time.perf_counter()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        if step_idx == 1 or step_idx % max(1, log_interval) == 0:
            running_loss = total_loss / max(1, total_samples)
            running_acc = total_correct / max(1, total_samples)
            print(
                f"[CIDS_TRAIN_STEP] step={step_idx} "
                f"loss={running_loss:.4f} "
                f"acc={running_acc:.4f}",
                flush=True,
            )
        if profiler is not None:
            profiler.step()
        stage_times["compute_time_sec"] += time.perf_counter() - compute_suffix_start_time
        if max_iters is not None and step_idx >= max_iters:
            break

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc, steps_ran, epoch_pairs, stage_times


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, image_preprocessor, cached_batch_loader=None):
    # 最小验证循环：只统计 loss 和 top-1 accuracy。
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step_idx, raw_batch in enumerate(dataloader, start=1):
        if cached_batch_loader is not None:
            images, labels = cached_batch_loader.fetch_batch(raw_batch)
        else:
            images, labels = _extract_images_and_labels_from_raw_batch(raw_batch)
        images = images.to(device, non_blocking=True)
        images = image_preprocessor.eval(images)
        labels = _move_labels_to_device(labels, device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        if step_idx == 1 or step_idx % 20 == 0:
            running_loss = total_loss / max(1, total_samples)
            running_acc = total_correct / max(1, total_samples)
            print(
                f"[CIDS_VAL_STEP] step={step_idx} "
                f"loss={running_loss:.4f} "
                f"acc={running_acc:.4f}",
                flush=True,
            )
    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def main():
    preprocess_time_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="最小 CIDS + ResNet18 训练脚本")
    parser.add_argument("--train-root", required=True, help="prepared train dataset 目录")
    parser.add_argument("--val-root", default=None, help="prepared val dataset 目录，可选")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--shuffle", choices=["0", "1"], default="1", help="训练集是否 shuffle")
    parser.add_argument("--max-train-iters", type=int, default=0, help="每个 epoch 最多跑多少个 train iter；0 表示不限制")
    parser.add_argument("--run-val", choices=["0", "1"], default="1", help="是否在每个 epoch 后运行验证")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--prefetch-depth", type=int, default=4, help="CIDS 预取深度")
    parser.add_argument("--cache-size", type=int, default=10, help="CIDS/BaM cache 大小，单位 MB")
    parser.add_argument("--registered-split", type=int, default=1, help="registered 模式下一个训练 batch 拆成多少个 sub-request")
    parser.add_argument(
        "--registered-skip-front",
        choices=["0", "1"],
        default=None,
        help="registered 模式是否开启 skip-front 预热；不显式传时沿用环境变量或默认值",
    )
    parser.add_argument("--ctrl-idx", type=int, default=0, help="使用的 GPU/controller 索引")
    parser.add_argument("--pretrained", action="store_true", help="是否使用 torchvision 预训练权重")
    parser.add_argument(
        "--io-mode",
        choices=["sync", "registered", "torch"],
        default="registered",
        help="读取模式：sync 为最原始同步读取，registered 为异步 registered try-service，torch 为 PyTorch 原生 DataLoader",
    )
    parser.add_argument(
        "--torch-read-mode",
        choices=["mmap", "buffered"],
        default="mmap",
        help="torch 分支的 prepared 文件读取方式：mmap 为内存映射，buffered 为整块读入内存",
    )
    parser.add_argument(
        "--enable-profile",
        choices=["0", "1"],
        default=os.environ.get("CIDS_ENABLE_PROFILE", "0"),
        help="是否开启和 GIDS 类似的 torch.profiler",
    )
    parser.add_argument(
        "--profile-dir",
        default=os.environ.get("CIDS_PROFILE_DIR", "./cids_profile"),
        help="profiler 输出目录",
    )
    parser.add_argument(
        "--enable-sample-cache",
        choices=["0", "1"],
        default="0",
        help="是否启用第一阶段 host sample cache；默认关闭，不影响原逻辑。",
    )
    parser.add_argument(
        "--sample-cache-capacity",
        type=int,
        default=4096,
        help="sample cache 最多缓存多少个 sample。",
    )
    parser.add_argument(
        "--sample-cache-pin-memory",
        choices=["0", "1"],
        default="1",
        help="sample cache 是否把 host tensor 放到 pinned memory。",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="训练日志输出间隔（按 iter 计）。",
    )
    parser.add_argument(
        "--enable-sample-importance",
        choices=["0", "1"],
        default="0",
        help="是否记录第一阶段 per-sample importance；默认关闭。",
    )
    parser.add_argument(
        "--enable-bam-policy-cache",
        choices=["0", "1"],
        default="0",
        help="是否启用独立的 BaM score-aware cache policy；不使用 CPU sample cache。",
    )
    parser.add_argument(
        "--importance-ema-alpha",
        type=float,
        default=0.9,
        help="per-sample importance 的 EMA 系数。",
    )
    parser.add_argument(
        "--importance-topk",
        type=int,
        default=5,
        help="日志里输出多少个 top-k importance sample。",
    )
    parser.add_argument(
        "--pads-rep-factor",
        type=float,
        default=1.5,
        help="最小版 PADS 的基础 hit 重复因子。",
    )
    parser.add_argument(
        "--pads-adaptive",
        choices=["0", "1"],
        default="1",
        help="是否根据 hit_rate / train_loss 轻量自适应调整下一轮 PADS 重复强度。",
    )
    parser.add_argument(
        "--pads-strategy",
        choices=["replace", "shade", "hotset"],
        default="replace",
        help="PADS resident 调度策略：replace 为轻量替换，shade 为 locality replay，hotset 为逻辑热集混合采样。",
    )
    parser.add_argument(
        "--pads-bias-scale",
        type=float,
        default=-1.0,
        help="PADS 控制参数；replace 下表示轻偏置缩放，shade 下表示 replay 缩放因子。",
    )
    parser.add_argument(
        "--pads-max-replace-fraction",
        type=float,
        default=-1.0,
        help="PADS 替换上限；小于 0 时按策略和 io-mode 选择默认值。",
    )
    args = parser.parse_args()
    max_train_iters = args.max_train_iters if args.max_train_iters > 0 else None
    run_val = args.run_val == "1"
    train_shuffle = args.shuffle == "1"
    enable_sample_cache = args.enable_sample_cache == "1"
    enable_sample_importance = args.enable_sample_importance == "1"
    enable_bam_policy_cache = args.enable_bam_policy_cache == "1"
    pads_adaptive = args.pads_adaptive == "1"
    enable_pads = enable_sample_importance and (enable_sample_cache or enable_bam_policy_cache)

    if enable_sample_cache and args.io_mode == "torch":
        raise ValueError("第一阶段 sample cache 只支持 CIDS sync/registered，不支持 torch 分支")
    if enable_sample_cache and enable_bam_policy_cache:
        raise ValueError(
            "当前简化实现里，CPU sample cache 和 BaM policy cache 先保持互斥，避免两套策略混用"
        )
    if enable_sample_importance and not (enable_sample_cache or enable_bam_policy_cache):
        raise ValueError("开启 sample importance 时，需要配合 sample cache 或 BaM policy cache")
    if enable_bam_policy_cache and args.io_mode == "torch":
        raise ValueError("BaM policy 只支持 CIDS sync/registered，不支持 torch 分支")

    if args.registered_skip_front is not None:
        os.environ["CIDS_REGISTERED_ENABLE_SKIP_FRONT"] = args.registered_skip_front
    os.environ["CIDS_REGISTERED_SPLIT"] = str(max(1, args.registered_split))

    device = torch.device(f"cuda:{args.ctrl_idx}" if torch.cuda.is_available() else "cpu")

    train_meta_time_start = time.perf_counter()
    train_meta = _load_meta(args.train_root)
    num_classes = len(train_meta.get("classes", []))
    if num_classes <= 0:
        raise ValueError("prepared train dataset 的 meta.json 中没有有效 classes 信息")
    image_size = int(train_meta["shape"][-1])
    cids_page_size, sample_bytes = _resnet18_sample_page_size(train_meta)
    train_meta_time_sec = time.perf_counter() - train_meta_time_start

    train_cids = None
    sample_cache = None
    cached_train_loader = None
    cached_val_loader = None
    importance_tracker = None
    bam_policy = None
    pads_planner = None
    pads_rep_factor = max(1.0, float(args.pads_rep_factor))
    pads_strategy = str(args.pads_strategy)
    # planner 参数：
    # - replace: 旧的 epoch 级轻替换
    # - shade: 更接近 SHADE 的 resident replay + locality 聚簇
    # - hotset: 逻辑 importance/hotset 主导，resident 只做低频校准
    if args.pads_bias_scale >= 0.0:
        pads_bias_scale = float(args.pads_bias_scale)
    elif pads_strategy == "hotset":
        pads_bias_scale = 0.0625 if args.io_mode == "registered" else 0.10
    elif pads_strategy == "shade":
        pads_bias_scale = 0.12 if args.io_mode == "registered" else 0.18
    else:
        pads_bias_scale = 0.15 if args.io_mode == "registered" else 0.25
    if args.pads_max_replace_fraction >= 0.0:
        pads_max_replace_fraction = float(args.pads_max_replace_fraction)
    elif pads_strategy == "hotset":
        pads_max_replace_fraction = 0.008 if args.io_mode == "registered" else 0.015
    elif pads_strategy == "shade":
        pads_max_replace_fraction = 0.002 if args.io_mode == "registered" else 0.004
    else:
        pads_max_replace_fraction = 0.005 if args.io_mode == "registered" else 0.01
    if enable_sample_importance:
        importance_tracker = SampleImportanceTracker(ema_alpha=args.importance_ema_alpha)
    if enable_bam_policy_cache:
        bam_policy = GPUBAMPolicy(
            num_pages=int(train_meta["num_samples"]),
            device=device,
            ema_alpha=args.importance_ema_alpha,
        )
    if enable_pads:
        if pads_strategy == "hotset":
            pads_planner = LogicalHotsetPADSPlanner(
                hot_fraction=pads_bias_scale,
                max_replace_fraction=pads_max_replace_fraction,
                hotset_size_scale=1.0,
                locality_window_batches=4 if args.io_mode == "registered" else 2,
            )
        elif pads_strategy == "shade":
            pads_planner = ShadePADSPlanner(
                replay_bias_scale=pads_bias_scale,
                max_replace_fraction=pads_max_replace_fraction,
                locality_window=16 if args.io_mode == "registered" else 8,
            )
        else:
            pads_planner = BAMPADSPlanner(
                replacement_bias_scale=pads_bias_scale,
                max_replace_fraction=pads_max_replace_fraction,
            )
    cids_init_time_sec = 0.0
    loader_init_time_start = time.perf_counter()
    if args.io_mode == "torch":
        _, train_loader = _build_torch_loader(
            prepared_root=args.train_root,
            batch_size=args.batch_size,
            shuffle=train_shuffle,
            torch_read_mode=args.torch_read_mode,
        )
        val_loader = None
        if run_val and args.val_root is not None:
            _, val_loader = _build_torch_loader(
                prepared_root=args.val_root,
                batch_size=args.batch_size,
                shuffle=False,
                torch_read_mode=args.torch_read_mode,
            )
    else:
        cids_init_time_start = time.perf_counter()
        train_cids = CIDS.from_prepared_dataset(
            args.train_root,
            ctrl_idx=args.ctrl_idx,
            cache_size=args.cache_size,
            page_size=cids_page_size,
        )
        if enable_bam_policy_cache:
            train_cids.enable_bam_policy_cache()
        cids_init_time_sec = time.perf_counter() - cids_init_time_start

        if enable_sample_cache:
            sample_cache = LRUSampleCache(
                capacity=args.sample_cache_capacity,
                pin_memory=(args.sample_cache_pin_memory == "1"),
                importance_tracker=importance_tracker,
            )
            cached_train_loader = CachedCIDSBatchLoader(
                cids_loader=train_cids,
                sample_cache=sample_cache,
                io_mode=args.io_mode,
                device=device,
            )
            _, train_loader = _build_index_loader(
                prepared_root=args.train_root,
                batch_size=args.batch_size,
                shuffle=train_shuffle,
                start_sample_id=0,
            )

            val_loader = None
            if run_val and args.val_root is not None:
                val_start_sample_id = int(train_meta["num_samples"])
                cached_val_loader = CachedCIDSBatchLoader(
                    cids_loader=train_cids,
                    sample_cache=sample_cache,
                    io_mode=args.io_mode,
                    device=device,
                )
                _, val_loader = _build_index_loader(
                    prepared_root=args.val_root,
                    batch_size=args.batch_size,
                    shuffle=False,
                    start_sample_id=val_start_sample_id,
                )
        else:
            _, train_loader = _build_loader(
                prepared_root=args.train_root,
                batch_size=args.batch_size,
                shuffle=train_shuffle,
                cids_loader=train_cids,
                prefetch_depth=args.prefetch_depth,
                start_sample_id=0,
                io_mode=args.io_mode,
                registered_split=args.registered_split,
            )

            val_loader = None
            if run_val and args.val_root is not None:
                val_start_sample_id = int(train_meta["num_samples"])
                _, val_loader = _build_loader(
                    prepared_root=args.val_root,
                    batch_size=args.batch_size,
                    shuffle=False,
                    cids_loader=train_cids,
                    prefetch_depth=args.prefetch_depth,
                    start_sample_id=val_start_sample_id,
                    io_mode=args.io_mode,
                    registered_split=args.registered_split,
                )
    loader_init_time_sec = time.perf_counter() - loader_init_time_start

    model_init_time_start = time.perf_counter()
    model = build_resnet18(
        num_classes=num_classes,
        pretrained=args.pretrained,
        in_channels=train_meta["shape"][0],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    image_preprocessor = ImageNetBatchPreprocessor(source_image_size=image_size)
    model_init_time_sec = time.perf_counter() - model_init_time_start

    print(
        f"[CIDS_TRAIN] device={device} num_classes={num_classes} "
        f"shape={train_meta['shape']} dtype={train_meta['dtype']} "
        f"io_mode={args.io_mode} "
        f"page_size={cids_page_size}B "
        f"sample_bytes={sample_bytes}B "
        f"cache_size={args.cache_size}MB "
        f"max_train_iters={max_train_iters if max_train_iters is not None else 'full'} "
        f"shuffle={train_shuffle} "
        f"run_val={run_val} "
        f"prefetch_depth={args.prefetch_depth} "
        f"registered_split={args.registered_split} "
        f"enable_sample_cache={enable_sample_cache} "
        f"sample_cache_capacity={args.sample_cache_capacity} "
        f"enable_sample_importance={enable_sample_importance} "
        f"enable_bam_policy_cache={enable_bam_policy_cache} "
        f"CIDS_REGISTERED_ENABLE_SKIP_FRONT={os.environ.get('CIDS_REGISTERED_ENABLE_SKIP_FRONT', '1')} "
        f"CIDS_PROFILE_GPU_TIMING={os.environ.get('CIDS_PROFILE_GPU_TIMING', '0')}",
        flush=True,
    )
    if args.io_mode == "torch":
        print(
            "[CIDS_TRAIN] 当前使用 PyTorch 原生 DataLoader，"
            f"直接读取 prepared dataset 文件，torch_read_mode={args.torch_read_mode}。",
            flush=True,
        )
    else:
        print(
            "[CIDS_TRAIN] 假设 prepared images 已经通过单独脚本写入 BaM；"
            "训练阶段不再负责写入。",
            flush=True,
        )
        if enable_sample_cache:
            print(
                f"[CIDS_CACHE] enabled=1 capacity={args.sample_cache_capacity} "
                f"pin_memory={args.sample_cache_pin_memory}",
                flush=True,
            )
        if enable_sample_importance:
            print(
                f"[CIDS_IMPORTANCE] enabled=1 ema_alpha={args.importance_ema_alpha} "
                f"topk={args.importance_topk}",
                flush=True,
            )
        if enable_pads:
            pads_sampler = "bam_resident" if enable_bam_policy_cache else "sample_cache"
            print(
                f"[CIDS_PADS] enabled=1 sampler={pads_sampler} base_rep_factor={pads_rep_factor:.2f} "
                f"strategy={pads_strategy} "
                f"adaptive={int(pads_adaptive)} "
                f"bias_scale={pads_bias_scale:.3f} "
                f"max_replace_fraction={pads_max_replace_fraction:.4f}",
                flush=True,
            )
        if enable_bam_policy_cache:
            print("[CIDS_BAM_POLICY] enabled=1 mode=score-aware-eviction", flush=True)

    preprocess_time_sec = time.perf_counter() - preprocess_time_start
    preprocess_other_time_sec = max(
        0.0,
        preprocess_time_sec
        - train_meta_time_sec
        - cids_init_time_sec
        - loader_init_time_sec
        - model_init_time_sec,
    )
    print(
        f"[CIDS_PREPROCESS_SUMMARY] preprocess_time_sec={preprocess_time_sec:.4f}",
        flush=True,
    )
    print(
        f"[CIDS_PREPROCESS_BREAKDOWN] meta_time_sec={train_meta_time_sec:.4f} "
        f"cids_init_time_sec={cids_init_time_sec:.4f} "
        f"loader_init_time_sec={loader_init_time_sec:.4f} "
        f"model_init_time_sec={model_init_time_sec:.4f} "
        f"other_time_sec={preprocess_other_time_sec:.4f}",
        flush=True,
    )

    profiler = _start_profiler(
        enabled=(args.enable_profile == "1"),
        output_dir=args.profile_dir,
    )
    total_train_iters = 0
    prev_train_loss = None
    total_stage_times = _zero_stage_times()
    train_time_start = time.perf_counter()
    try:
        for epoch in range(args.epochs):
            epoch_stage_times = _zero_stage_times()
            submit_time_before = train_cids.GIDS_submit_time if train_cids is not None else 0.0
            poll_time_before = train_cids.GIDS_poll_time if train_cids is not None else 0.0
            get_time_before = train_cids.GIDS_get_time if train_cids is not None else 0.0
            wait_time_before = train_cids.GIDS_wait_time if train_cids is not None else 0.0
            train_loss, train_acc, epoch_train_iters, epoch_pairs, train_stage_times = train_one_epoch(
                model, train_loader, optimizer, criterion, device, image_preprocessor,
                profiler=profiler, max_iters=max_train_iters,
                cached_batch_loader=cached_train_loader,
                log_interval=args.log_interval,
                importance_tracker=importance_tracker,
                bam_policy=bam_policy,
                bam_policy_cids=train_cids,
                global_step_offset=total_train_iters,
                collect_epoch_pairs=enable_pads,
            )
            total_train_iters += epoch_train_iters
            submit_time = train_cids.GIDS_submit_time if train_cids is not None else 0.0
            poll_time = train_cids.GIDS_poll_time if train_cids is not None else 0.0
            get_time = train_cids.GIDS_get_time if train_cids is not None else 0.0
            wait_time = train_cids.GIDS_wait_time if train_cids is not None else 0.0
            submit_time_delta = max(0.0, submit_time - submit_time_before)
            poll_time_delta = max(0.0, poll_time - poll_time_before)
            get_time_delta = max(0.0, get_time - get_time_before)
            wait_time_delta = max(0.0, wait_time - wait_time_before)
            if args.io_mode == "registered":
                epoch_stage_times["route_time_sec"] += submit_time_delta
                epoch_stage_times["io_stall_time_sec"] += poll_time_delta
                epoch_stage_times["extract_time_sec"] += max(
                    0.0,
                    train_stage_times["extract_time_sec"] - submit_time_delta - poll_time_delta,
                )
            else:
                epoch_stage_times["extract_time_sec"] += max(
                    0.0,
                    train_stage_times["extract_time_sec"] - wait_time_delta,
                )
                epoch_stage_times["io_stall_time_sec"] += wait_time_delta
            epoch_stage_times["compute_time_sec"] += train_stage_times["compute_time_sec"]
            epoch_stage_times["apply_time_sec"] += train_stage_times["apply_time_sec"]
            print(
                f"[CIDS_TRAIN] epoch={epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"train_iters={epoch_train_iters} "
                f"submit_time={submit_time:.4f} "
                f"wait_time={wait_time:.4f}",
                flush=True,
            )
            if cached_train_loader is not None:
                print(
                    f"[CIDS_CACHE] epoch={epoch + 1}/{args.epochs} "
                    f"{_format_cache_summary(cached_train_loader.cache_summary())}",
                    flush=True,
                )
            if importance_tracker is not None:
                print(
                    f"[CIDS_IMPORTANCE] epoch={epoch + 1}/{args.epochs} "
                    f"{_format_importance_summary(importance_tracker.summary(args.importance_topk))}",
                    flush=True,
                )
            if bam_policy is not None:
                print(
                    f"[CIDS_BAM_POLICY] epoch={epoch + 1}/{args.epochs} "
                    f"{_format_bam_policy_summary(bam_policy.summary())}",
                    flush=True,
                )

            if val_loader is not None:
                val_loss, val_acc = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    image_preprocessor,
                    cached_batch_loader=cached_val_loader,
                )
                print(
                    f"[CIDS_TRAIN] epoch={epoch + 1}/{args.epochs} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                    flush=True,
                )
            if train_cids is not None:
                print(
                    f"[CIDS_BAM_CACHE] epoch={epoch + 1}/{args.epochs} begin",
                    flush=True,
                )
                train_cids.BAM_FS.print_stats()
                print(
                    f"[CIDS_BAM_CACHE] epoch={epoch + 1}/{args.epochs} end",
                    flush=True,
                )
            if enable_pads and epoch + 1 < args.epochs:
                pads_stage_start_time = time.perf_counter()
                resident_lookup = None
                if enable_bam_policy_cache:
                    resident_lookup = _build_bam_resident_lookup(
                        epoch_pairs=epoch_pairs,
                        cids_loader=train_cids,
                    )
                elif enable_sample_cache:
                    resident_lookup = _build_sample_cache_resident_lookup(
                        epoch_pairs=epoch_pairs,
                        sample_cache=sample_cache,
                    )

                observed_rate = _resident_rate(epoch_pairs, resident_lookup)
                effective_rep_factor, pads_mode = _choose_pads_rep_factor(
                    base_rep_factor=pads_rep_factor,
                    adaptive_enabled=pads_adaptive,
                    observed_rate=observed_rate,
                    prev_train_loss=prev_train_loss,
                    curr_train_loss=train_loss,
                )
                if enable_bam_policy_cache and pads_planner is not None:
                    planner_kwargs = dict(
                        epoch_pairs=epoch_pairs,
                        resident_lookup=resident_lookup,
                        importance_tracker=importance_tracker,
                        rep_factor=effective_rep_factor,
                        shuffle_seed=epoch + 1,
                        epoch_index=epoch + 1,
                    )
                    if pads_strategy == "hotset":
                        planned_pairs, pads_summary = pads_planner.build_next_epoch_plan(
                            batch_size=args.batch_size,
                            **planner_kwargs,
                        )
                    else:
                        planned_pairs, pads_summary = pads_planner.build_next_epoch_plan(
                            **planner_kwargs
                        )
                else:
                    planned_pairs, pads_summary = _build_pads_next_epoch_plan(
                        epoch_pairs=epoch_pairs,
                        resident_lookup=resident_lookup,
                        importance_tracker=importance_tracker,
                        rep_factor=effective_rep_factor,
                        shuffle_seed=epoch + 1,
                        sampler_name=("bam_resident" if enable_bam_policy_cache else "sample_cache"),
                    )
                if planned_pairs is not None and pads_summary is not None:
                    pads_summary["base_rep_factor"] = float(pads_rep_factor)
                    pads_summary["effective_rep_factor"] = float(effective_rep_factor)
                    pads_summary["mode"] = pads_mode
                    pads_summary["strategy"] = pads_strategy
                    pads_summary["resident_rate"] = float(observed_rate)
                    pads_summary["base_size"] = len(epoch_pairs) if epoch_pairs is not None else 0
                    print(
                        f"[CIDS_PADS] epoch={epoch + 1}/{args.epochs} "
                        f"{_format_pads_summary(pads_summary)}",
                        flush=True,
                    )
                    if enable_bam_policy_cache:
                        _, train_loader = _build_planned_cids_loader(
                            epoch_pairs=planned_pairs,
                            batch_size=args.batch_size,
                            cids_loader=train_cids,
                            prefetch_depth=args.prefetch_depth,
                            io_mode=args.io_mode,
                            registered_split=args.registered_split,
                        )
                    else:
                        _, train_loader = _build_planned_index_loader(
                            epoch_pairs=planned_pairs,
                            batch_size=args.batch_size,
                        )
                pads_stage_elapsed = time.perf_counter() - pads_stage_start_time
                epoch_stage_times["route_time_sec"] += pads_stage_elapsed
            print(
                f"[CIDS_RECA] epoch={epoch + 1}/{args.epochs} "
                f"{_format_stage_times(epoch_stage_times)}",
                flush=True,
            )
            _merge_stage_times(total_stage_times, epoch_stage_times)
            prev_train_loss = train_loss
    finally:
        _stop_profiler(profiler)
        if cached_train_loader is not None:
            print(
                f"[CIDS_CACHE_SUMMARY] {_format_cache_summary(cached_train_loader.cache_summary())}",
                flush=True,
            )
        if importance_tracker is not None:
            print(
                f"[CIDS_IMPORTANCE_SUMMARY] "
                f"{_format_importance_summary(importance_tracker.summary(args.importance_topk))}",
                flush=True,
            )
        if bam_policy is not None:
            print(
                f"[CIDS_BAM_POLICY_SUMMARY] "
                f"{_format_bam_policy_summary(bam_policy.summary())}",
                flush=True,
            )
        total_train_time_sec = time.perf_counter() - train_time_start
        total_end_to_end_time_sec = preprocess_time_sec + total_train_time_sec
        print(
            f"[CIDS_TRAIN_SUMMARY] preprocess_time_sec={preprocess_time_sec:.4f} "
            f"meta_time_sec={train_meta_time_sec:.4f} "
            f"cids_init_time_sec={cids_init_time_sec:.4f} "
            f"loader_init_time_sec={loader_init_time_sec:.4f} "
            f"model_init_time_sec={model_init_time_sec:.4f} "
            f"other_time_sec={preprocess_other_time_sec:.4f} "
            f"total_train_time_sec={total_train_time_sec:.4f} "
            f"total_end_to_end_time_sec={total_end_to_end_time_sec:.4f} "
            f"total_train_iters={total_train_iters}",
            flush=True,
        )
        print(
            f"[CIDS_RECA_SUMMARY] {_format_stage_times(total_stage_times)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
