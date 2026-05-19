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
from cache_runtime import CachedCIDSBatchLoader, LRUSampleCache, SampleImportanceTracker


def _load_meta(prepared_root):
    # 读取 prepared dataset 的元信息，确定类别数和图片形状。
    meta_path = Path(prepared_root) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        self.torch_read_mode = str(torch_read_mode)
        self.labels = np.load(self.prepared_root / self.meta.get("labels_file", "labels.npy"))
        images_path = self.prepared_root / self.meta.get("images_file", "images.bin")
        if self.torch_read_mode == "mmap":
            self.images = np.memmap(
                images_path,
                dtype=self.dtype,
                mode="r",
                shape=(self.num_samples, self.sample_dim),
            )
        elif self.torch_read_mode == "buffered":
            self.images = np.fromfile(images_path, dtype=self.dtype).reshape(
                self.num_samples, self.sample_dim
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
        image_np = np.array(self.images[idx], copy=True)
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
    return sample_ids.detach().cpu()


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
    return labels.detach().cpu()


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


def _importance_score_for_sample(importance_tracker, sample_id):
    if importance_tracker is None:
        return 0.0
    state = importance_tracker.get(sample_id)
    if state is None:
        return 0.0
    return float(state.ema_score)


def _build_pads_next_epoch_plan(
    epoch_pairs,
    sample_cache,
    importance_tracker,
    rep_factor=1.5,
    shuffle_seed=0,
):
    if not epoch_pairs or sample_cache is None or importance_tracker is None:
        return None, None

    hit_pairs = []
    miss_pairs = []
    for sample_id, label in epoch_pairs:
        pair = (int(sample_id), int(label))
        if sample_cache.contains(sample_id):
            hit_pairs.append(pair)
        else:
            miss_pairs.append(pair)

    if not hit_pairs or not miss_pairs:
        summary = {
            "base_rep_factor": float(rep_factor),
            "effective_rep_factor": float(rep_factor),
            "mode": "base",
            "hit_rate": 0.0,
            "base_size": len(epoch_pairs),
            "hit_count": len(hit_pairs),
            "miss_count": len(miss_pairs),
            "boosted_hits": 0,
            "planned_size": len(epoch_pairs),
            "sampler": "basic",
        }
        return list(epoch_pairs), summary

    ranked_hits = sorted(
        hit_pairs,
        key=lambda pair: _importance_score_for_sample(importance_tracker, pair[0]),
        reverse=True,
    )
    extra_hit_budget = min(
        len(miss_pairs),
        max(0, int(round(len(hit_pairs) * max(0.0, float(rep_factor) - 1.0)))),
    )
    boosted_hits = ranked_hits[:extra_hit_budget]
    hit_schedule = list(ranked_hits) + list(boosted_hits)
    kept_misses = miss_pairs[: max(0, len(miss_pairs) - extra_hit_budget)]

    rng = random.Random(int(shuffle_seed))
    rng.shuffle(hit_schedule)
    rng.shuffle(kept_misses)

    planned_pairs = hit_schedule + kept_misses
    summary = {
        "base_rep_factor": float(rep_factor),
        "effective_rep_factor": float(rep_factor),
        "mode": "base",
        "hit_rate": len(hit_pairs) / max(1, len(epoch_pairs)),
        "base_size": len(epoch_pairs),
        "hit_count": len(hit_pairs),
        "miss_count": len(miss_pairs),
        "boosted_hits": len(boosted_hits),
        "planned_size": len(planned_pairs),
        "sampler": "basic",
    }
    return planned_pairs, summary


def _format_pads_summary(summary):
    return (
        f"sampler={summary.get('sampler', 'basic')} "
        f"base_rep_factor={summary.get('base_rep_factor', 0.0):.2f} "
        f"effective_rep_factor={summary.get('effective_rep_factor', 0.0):.2f} "
        f"mode={summary.get('mode', 'base')} "
        f"hit_rate={summary.get('hit_rate', 0.0):.4f} "
        f"base_size={summary.get('base_size', 0)} "
        f"hit_count={summary.get('hit_count', 0)} "
        f"miss_count={summary.get('miss_count', 0)} "
        f"boosted_hits={summary.get('boosted_hits', 0)} "
        f"planned_size={summary.get('planned_size', 0)}"
    )


def _cache_hit_rate(summary):
    if not summary:
        return 0.0
    hits = int(summary.get("hits", 0))
    misses = int(summary.get("misses", 0))
    total = hits + misses
    return (hits / total) if total > 0 else 0.0


def _choose_pads_rep_factor(
    base_rep_factor,
    adaptive_enabled,
    cache_summary,
    prev_train_loss,
    curr_train_loss,
):
    base_rep_factor = max(1.0, float(base_rep_factor))
    if not adaptive_enabled:
        return base_rep_factor, "fixed"

    hit_rate = _cache_hit_rate(cache_summary)
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

    if hit_rate < 0.30 or loss_plateau:
        rep_factor += 0.25
        mode = "aggressive"
    elif hit_rate > 0.45 and loss_improving:
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
    global_step_offset=0,
    collect_epoch_pairs=False,
):
    # 最小训练循环：只做前向、反向、优化和简单精度统计。
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    steps_ran = 0
    epoch_pairs = [] if collect_epoch_pairs else None

    for step_idx, raw_batch in enumerate(dataloader, start=1):
        steps_ran = step_idx
        sample_ids = None
        raw_labels = None
        if importance_tracker is not None:
            sample_ids = _extract_sample_ids_from_raw_batch(raw_batch)
            if collect_epoch_pairs:
                raw_labels = _extract_labels_from_raw_batch(raw_batch)
        if cached_batch_loader is not None:
            images, labels = cached_batch_loader.fetch_batch(raw_batch)
        else:
            images, labels = raw_batch
        images = images.to(device, non_blocking=True)
        images = image_preprocessor.train(images)
        labels = _move_labels_to_device(labels, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        sample_losses = F.cross_entropy(logits, labels, reduction="none")
        loss = sample_losses.mean()
        if importance_tracker is not None and sample_ids is not None:
            sample_scores = _compute_batch_rank_scores(sample_losses)
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
                        sample_ids.view(-1).tolist(),
                        raw_labels.view(-1).tolist(),
                    )
                )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        if step_idx == 1 or step_idx % max(1, log_interval) == 0:
            print(
                f"[CIDS_TRAIN_STEP] step={step_idx} "
                f"loss={(total_loss / max(1, total_samples)):.4f} "
                f"acc={(total_correct / max(1, total_samples)):.4f}",
                flush=True,
            )
        if profiler is not None:
            profiler.step()
        if max_iters is not None and step_idx >= max_iters:
            break

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc, steps_ran, epoch_pairs


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
            images, labels = raw_batch
        images = images.to(device, non_blocking=True)
        images = image_preprocessor.eval(images)
        labels = _move_labels_to_device(labels, device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        if step_idx == 1 or step_idx % 20 == 0:
            print(
                f"[CIDS_VAL_STEP] step={step_idx} "
                f"loss={(total_loss / max(1, total_samples)):.4f} "
                f"acc={(total_correct / max(1, total_samples)):.4f}",
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
        "--force-sync-read",
        choices=["0", "1"],
        default=None,
        help="是否强制底层 read_feature 走 GIDS_FORCE_SYNC_READ，同步模式默认自动设为 1",
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
    args = parser.parse_args()
    max_train_iters = args.max_train_iters if args.max_train_iters > 0 else None
    run_val = args.run_val == "1"
    train_shuffle = args.shuffle == "1"
    enable_sample_cache = args.enable_sample_cache == "1"
    enable_sample_importance = args.enable_sample_importance == "1"
    pads_adaptive = args.pads_adaptive == "1"

    if enable_sample_cache and args.io_mode == "torch":
        raise ValueError("第一阶段 sample cache 只支持 CIDS sync/registered，不支持 torch 分支")
    if enable_sample_importance and not enable_sample_cache:
        raise ValueError("当前简化实现里，sample importance 依赖保留 sample_id 的 cache/index 路径，请先开启 sample cache")

    if args.force_sync_read is not None:
        os.environ["GIDS_FORCE_SYNC_READ"] = args.force_sync_read
    elif args.io_mode == "sync":
        os.environ["GIDS_FORCE_SYNC_READ"] = "1"
    else:
        os.environ.setdefault("GIDS_FORCE_SYNC_READ", "0")

    if args.io_mode == "sync" and os.environ.get("GIDS_FORCE_SYNC_READ", "0") != "1":
        raise ValueError(
            "sync 模式必须配合 GIDS_FORCE_SYNC_READ=1，"
            "否则底层会误走旧的 async read_feature 路径。"
        )

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
    train_meta_time_sec = time.perf_counter() - train_meta_time_start

    train_cids = None
    sample_cache = None
    cached_train_loader = None
    cached_val_loader = None
    importance_tracker = None
    pads_rep_factor = max(1.0, float(args.pads_rep_factor))
    if enable_sample_importance:
        importance_tracker = SampleImportanceTracker(ema_alpha=args.importance_ema_alpha)
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
        )
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
        f"GIDS_FORCE_SYNC_READ={os.environ.get('GIDS_FORCE_SYNC_READ', '0')} "
        f"cache_size={args.cache_size}MB "
        f"max_train_iters={max_train_iters if max_train_iters is not None else 'full'} "
        f"shuffle={train_shuffle} "
        f"run_val={run_val} "
        f"prefetch_depth={args.prefetch_depth} "
        f"registered_split={args.registered_split} "
        f"enable_sample_cache={enable_sample_cache} "
        f"sample_cache_capacity={args.sample_cache_capacity} "
        f"enable_sample_importance={enable_sample_importance} "
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
            print(
                f"[CIDS_PADS] enabled=1 base_rep_factor={pads_rep_factor:.2f} "
                f"adaptive={int(pads_adaptive)}",
                flush=True,
            )

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
    train_time_start = time.perf_counter()
    try:
        for epoch in range(args.epochs):
            train_loss, train_acc, epoch_train_iters, epoch_pairs = train_one_epoch(
                model, train_loader, optimizer, criterion, device, image_preprocessor,
                profiler=profiler, max_iters=max_train_iters,
                cached_batch_loader=cached_train_loader,
                log_interval=args.log_interval,
                importance_tracker=importance_tracker,
                global_step_offset=total_train_iters,
                collect_epoch_pairs=(enable_sample_cache and enable_sample_importance),
            )
            total_train_iters += epoch_train_iters
            submit_time = train_cids.GIDS_submit_time if train_cids is not None else 0.0
            wait_time = train_cids.GIDS_wait_time if train_cids is not None else 0.0
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
            if enable_sample_cache and enable_sample_importance and epoch + 1 < args.epochs:
                cache_summary = (
                    cached_train_loader.cache_summary()
                    if cached_train_loader is not None
                    else None
                )
                effective_rep_factor, pads_mode = _choose_pads_rep_factor(
                    base_rep_factor=pads_rep_factor,
                    adaptive_enabled=pads_adaptive,
                    cache_summary=cache_summary,
                    prev_train_loss=prev_train_loss,
                    curr_train_loss=train_loss,
                )
                planned_pairs, pads_summary = _build_pads_next_epoch_plan(
                    epoch_pairs=epoch_pairs,
                    sample_cache=sample_cache,
                    importance_tracker=importance_tracker,
                    rep_factor=effective_rep_factor,
                    shuffle_seed=epoch + 1,
                )
                if planned_pairs is not None and pads_summary is not None:
                    pads_summary["base_rep_factor"] = float(pads_rep_factor)
                    pads_summary["effective_rep_factor"] = float(effective_rep_factor)
                    pads_summary["mode"] = pads_mode
                    pads_summary["hit_rate"] = _cache_hit_rate(cache_summary)
                    print(
                        f"[CIDS_PADS] epoch={epoch + 1}/{args.epochs} "
                        f"{_format_pads_summary(pads_summary)}",
                        flush=True,
                    )
                    _, train_loader = _build_planned_index_loader(
                        epoch_pairs=planned_pairs,
                        batch_size=args.batch_size,
                    )
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


if __name__ == "__main__":
    main()
