import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.profiler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
GIDS_SETUP_ROOT = REPO_ROOT / "GIDS_Setup"
if str(GIDS_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(GIDS_SETUP_ROOT))
if str(REPO_ROOT / "cnn_evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "cnn_evaluation"))

from GIDS import CIDS, CIDSPreparedDataset, CIDS_DataLoader
from cids_alexnet import build_alexnet
from cids_image_transforms import ImageNetBatchPreprocessor


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


def _move_labels_to_device(labels, device):
    # CIDS 当前返回的是 (images, labels) 形式，这里统一把 label 放到训练设备。
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    return labels.to(device, non_blocking=True)


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
):
    # 最小训练循环：只做前向、反向、优化和简单精度统计。
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    steps_ran = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for step_idx, (images, labels) in enumerate(progress, start=1):
        steps_ran = step_idx
        images = images.to(device, non_blocking=True)
        images = image_preprocessor.train(images)
        labels = _move_labels_to_device(labels, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        progress.set_postfix(
            loss=f"{(total_loss / max(1, total_samples)):.4f}",
            acc=f"{(total_correct / max(1, total_samples)):.4f}",
        )
        if profiler is not None:
            profiler.step()
        if max_iters is not None and step_idx >= max_iters:
            break

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc, steps_ran


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, image_preprocessor):
    # 最小验证循环：只统计 loss 和 top-1 accuracy。
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(dataloader, desc="val", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        images = image_preprocessor.eval(images)
        labels = _move_labels_to_device(labels, device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        progress.set_postfix(
            loss=f"{(total_loss / max(1, total_samples)):.4f}",
            acc=f"{(total_correct / max(1, total_samples)):.4f}",
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
    args = parser.parse_args()
    max_train_iters = args.max_train_iters if args.max_train_iters > 0 else None
    run_val = args.run_val == "1"

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
    cids_init_time_sec = 0.0
    loader_init_time_start = time.perf_counter()
    if args.io_mode == "torch":
        _, train_loader = _build_torch_loader(
            prepared_root=args.train_root,
            batch_size=args.batch_size,
            shuffle=True,
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

        _, train_loader = _build_loader(
            prepared_root=args.train_root,
            batch_size=args.batch_size,
            shuffle=True,
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
    model = build_alexnet(
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
        f"run_val={run_val} "
        f"prefetch_depth={args.prefetch_depth} "
        f"registered_split={args.registered_split} "
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

    preprocess_time_sec = time.perf_counter() - preprocess_time_start
    print(
        f"[CIDS_PREPROCESS_SUMMARY] preprocess_time_sec={preprocess_time_sec:.4f}",
        flush=True,
    )
    preprocess_other_time_sec = max(
        0.0,
        preprocess_time_sec
        - train_meta_time_sec
        - cids_init_time_sec
        - loader_init_time_sec
        - model_init_time_sec,
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
    train_time_start = time.perf_counter()
    try:
        for epoch in tqdm(range(args.epochs)):
            train_loss, train_acc, epoch_train_iters = train_one_epoch(
                model, train_loader, optimizer, criterion, device, image_preprocessor,
                profiler=profiler, max_iters=max_train_iters)
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

            if val_loader is not None:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device, image_preprocessor)
                print(
                    f"[CIDS_TRAIN] epoch={epoch + 1}/{args.epochs} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                    flush=True,
                )
    finally:
        _stop_profiler(profiler)
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
