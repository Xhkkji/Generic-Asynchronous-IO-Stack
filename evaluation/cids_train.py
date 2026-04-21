import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
GIDS_SETUP_ROOT = REPO_ROOT / "GIDS_Setup"
if str(GIDS_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(GIDS_SETUP_ROOT))
if str(REPO_ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from GIDS import CIDS, CIDSPreparedDataset, CIDS_DataLoader
from cids_resnet18 import build_resnet18


def _load_meta(prepared_root):
    # 读取 prepared dataset 的元信息，确定类别数和图片形状。
    meta_path = Path(prepared_root) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_loader(prepared_root, batch_size, shuffle, cids_loader, prefetch_depth,
                  start_sample_id=0, io_mode="registered"):
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
    )
    return dataset, dataloader


def _move_labels_to_device(labels, device):
    # CIDS 当前返回的是 (images, labels) 形式，这里统一把 label 放到训练设备。
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    return labels.to(device, non_blocking=True)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    # 最小训练循环：只做前向、反向、优化和简单精度统计。
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = _move_labels_to_device(labels, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    # 最小验证循环：只统计 loss 和 top-1 accuracy。
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = _move_labels_to_device(labels, device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="最小 CIDS + ResNet18 训练脚本")
    parser.add_argument("--train-root", required=True, help="prepared train dataset 目录")
    parser.add_argument("--val-root", default=None, help="prepared val dataset 目录，可选")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--prefetch-depth", type=int, default=4, help="CIDS 预取深度")
    parser.add_argument("--ctrl-idx", type=int, default=0, help="使用的 GPU/controller 索引")
    parser.add_argument("--pretrained", action="store_true", help="是否使用 torchvision 预训练权重")
    parser.add_argument(
        "--io-mode",
        choices=["sync", "registered"],
        default="registered",
        help="读取模式：sync 为最原始同步读取，registered 为异步 registered try-service",
    )
    parser.add_argument(
        "--force-sync-read",
        choices=["0", "1"],
        default=None,
        help="是否强制底层 read_feature 走 GIDS_FORCE_SYNC_READ，同步模式默认自动设为 1",
    )
    args = parser.parse_args()

    if args.force_sync_read is not None:
        os.environ["GIDS_FORCE_SYNC_READ"] = args.force_sync_read
    elif args.io_mode == "sync":
        os.environ["GIDS_FORCE_SYNC_READ"] = "1"
    else:
        os.environ.setdefault("GIDS_FORCE_SYNC_READ", "0")

    device = torch.device(f"cuda:{args.ctrl_idx}" if torch.cuda.is_available() else "cpu")

    train_meta = _load_meta(args.train_root)
    num_classes = len(train_meta.get("classes", []))
    if num_classes <= 0:
        raise ValueError("prepared train dataset 的 meta.json 中没有有效 classes 信息")

    train_cids = CIDS.from_prepared_dataset(
        args.train_root,
        ctrl_idx=args.ctrl_idx,
    )

    _, train_loader = _build_loader(
        prepared_root=args.train_root,
        batch_size=args.batch_size,
        shuffle=True,
        cids_loader=train_cids,
        prefetch_depth=args.prefetch_depth,
        start_sample_id=0,
        io_mode=args.io_mode,
    )

    val_loader = None
    if args.val_root is not None:
        val_start_sample_id = int(train_meta["num_samples"])
        _, val_loader = _build_loader(
            prepared_root=args.val_root,
            batch_size=args.batch_size,
            shuffle=False,
            cids_loader=train_cids,
            prefetch_depth=args.prefetch_depth,
            start_sample_id=val_start_sample_id,
            io_mode=args.io_mode,
        )

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

    print(
        f"[CIDS_TRAIN] device={device} num_classes={num_classes} "
        f"shape={train_meta['shape']} dtype={train_meta['dtype']} "
        f"io_mode={args.io_mode} "
        f"GIDS_FORCE_SYNC_READ={os.environ.get('GIDS_FORCE_SYNC_READ', '0')}",
        flush=True,
    )
    print(
        "[CIDS_TRAIN] 假设 prepared images 已经通过单独脚本写入 BaM；"
        "训练阶段不再负责写入。",
        flush=True,
    )

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        print(
            f"[CIDS_TRAIN] epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"submit_time={train_cids.GIDS_submit_time:.4f} "
            f"wait_time={train_cids.GIDS_wait_time:.4f}",
            flush=True,
        )

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(
                f"[CIDS_TRAIN] epoch={epoch + 1}/{args.epochs} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
