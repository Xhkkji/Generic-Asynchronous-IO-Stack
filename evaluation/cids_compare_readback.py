import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GIDS_SETUP_ROOT = REPO_ROOT / "GIDS_Setup"
if str(GIDS_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(GIDS_SETUP_ROOT))

from GIDS import CIDS


def _load_meta(prepared_root):
    meta_path = Path(prepared_root) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_prepared_batch(prepared_root, sample_indices):
    # 直接从 prepared dataset 的 images.bin / labels.npy 读取一批样本，作为 torch 基线。
    meta = _load_meta(prepared_root)
    sample_shape = tuple(meta["shape"])
    sample_dim = int(meta["sample_dim"])
    num_samples = int(meta["num_samples"])
    dtype = np.float32

    labels = np.load(Path(prepared_root) / meta.get("labels_file", "labels.npy"))
    images = np.memmap(
        Path(prepared_root) / meta.get("images_file", "images.bin"),
        dtype=dtype,
        mode="r",
        shape=(num_samples, sample_dim),
    )

    batch_images = np.stack([np.array(images[idx], copy=True) for idx in sample_indices], axis=0)
    batch_labels = np.array([labels[idx] for idx in sample_indices], dtype=np.int64)

    batch_images = torch.from_numpy(batch_images).view(len(sample_indices), *sample_shape)
    batch_labels = torch.from_numpy(batch_labels)
    return batch_images, batch_labels


def _compare_tensors(torch_images, cids_images):
    # 输出两条读取路径的张量差异统计，便于快速定位布局/数值问题。
    diff = (cids_images - torch_images).abs()
    print(
        "[CIDS_COMPARE] torch "
        f"min={torch_images.min().item():.6f} "
        f"max={torch_images.max().item():.6f} "
        f"mean={torch_images.mean().item():.6f}",
        flush=True,
    )
    print(
        "[CIDS_COMPARE] cids  "
        f"min={cids_images.min().item():.6f} "
        f"max={cids_images.max().item():.6f} "
        f"mean={cids_images.mean().item():.6f}",
        flush=True,
    )
    print(
        "[CIDS_COMPARE] diff  "
        f"max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f}",
        flush=True,
    )

    flat_torch = torch_images.reshape(torch_images.size(0), -1)
    flat_cids = cids_images.reshape(cids_images.size(0), -1)
    for row in range(min(3, torch_images.size(0))):
        torch_head = flat_torch[row, :8].tolist()
        cids_head = flat_cids[row, :8].tolist()
        print(f"[CIDS_COMPARE] sample[{row}] torch[:8]={torch_head}", flush=True)
        print(f"[CIDS_COMPARE] sample[{row}] cids [:8]={cids_head}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="比较 torch prepared 直读和 CIDS sync 读回的张量差异")
    parser.add_argument("--train-root", required=True, help="prepared train dataset 目录")
    parser.add_argument("--val-root", default=None, help="prepared val dataset 目录，可选")
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="要比较哪一部分数据",
    )
    parser.add_argument("--ctrl-idx", type=int, default=0, help="GPU/controller 索引")
    parser.add_argument("--batch-size", type=int, default=4, help="一次比较多少个样本")
    parser.add_argument("--start-index", type=int, default=0, help="从 prepared dataset 的第几个样本开始比较")
    args = parser.parse_args()

    os.environ["GIDS_FORCE_SYNC_READ"] = "1"

    prepared_root = args.train_root if args.split == "train" else args.val_root
    if prepared_root is None:
        raise ValueError("比较 val 时必须提供 --val-root")

    meta = _load_meta(args.train_root)
    sample_offset = 0
    if args.split == "val":
        sample_offset = int(meta["num_samples"])

    sample_indices = list(range(args.start_index, args.start_index + args.batch_size))
    torch_images, torch_labels = _load_prepared_batch(prepared_root, sample_indices)

    cids = CIDS.from_prepared_dataset(
        args.train_root,
        ctrl_idx=args.ctrl_idx,
    )
    cids_sample_ids = torch.tensor(
        [sample_offset + idx for idx in sample_indices],
        dtype=torch.long,
    )
    batch = (cids_sample_ids, torch_labels.clone())
    cids_images, cids_labels = cids.fetch_samples_sync(batch, cids.cids_device)
    cids_images = cids_images.detach().cpu()
    cids_labels = cids_labels.detach().cpu()

    print(
        f"[CIDS_COMPARE] split={args.split} sample_indices={sample_indices} "
        f"sample_id_offset={sample_offset}",
        flush=True,
    )
    print(
        f"[CIDS_COMPARE] labels_equal={torch.equal(torch_labels, cids_labels)}",
        flush=True,
    )
    _compare_tensors(torch_images, cids_images)


if __name__ == "__main__":
    main()
