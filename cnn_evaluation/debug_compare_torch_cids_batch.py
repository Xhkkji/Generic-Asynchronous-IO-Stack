#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

# 调试脚本功能：
# - 允许从任意工作目录直接运行
# - 把仓库根目录加入 sys.path，便于导入 GIDS_Setup
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from GIDS_Setup.GIDS.CIDS import CIDS


def _load_meta(prepared_root):
    with open(Path(prepared_root) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_torch_batch(prepared_root, sample_ids):
    # Torch 直读功能：
    # - 直接从 images.bin / labels.npy 读取同一批 sample
    meta = _load_meta(prepared_root)
    shape = tuple(meta["shape"])
    sample_dim = int(meta["sample_dim"])
    sample_bytes = int(meta["sample_bytes"])
    dtype_name = str(meta["dtype"])
    dtype_map = {
        "uint8": np.uint8,
        "float16": np.float16,
        "float32": np.float32,
        "int64": np.int64,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype_name}")

    dtype = dtype_map[dtype_name]
    itemsize = np.dtype(dtype).itemsize
    storage_sample_dim = sample_bytes // itemsize

    labels = np.load(Path(prepared_root) / meta.get("labels_file", "labels.npy"))
    images = np.memmap(
        Path(prepared_root) / meta.get("images_file", "images.bin"),
        dtype=dtype,
        mode="r",
        shape=(int(meta["num_samples"]), storage_sample_dim),
    )

    batch_images = []
    batch_labels = []
    for sample_id in sample_ids:
        image_np = np.array(images[sample_id, :sample_dim], copy=True)
        image = torch.from_numpy(image_np).view(*shape)
        if dtype_name == "uint8":
            image = image.to(torch.float32).div_(255.0)
        elif image.dtype != torch.float32:
            image = image.to(torch.float32)
        batch_images.append(image)
        batch_labels.append(int(labels[sample_id]))

    return torch.stack(batch_images, dim=0), torch.tensor(batch_labels, dtype=torch.long)


def _extract_images_and_labels(batch):
    # CIDS 返回兼容功能：
    # - 兼容 tuple/list
    # - 兼容带 sample_ids 的 dict
    if isinstance(batch, dict):
        return batch["images"], batch["labels"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(f"unsupported batch type: {type(batch)!r}")


def _print_tensor_stats(name, tensor):
    tensor = tensor.detach().to(torch.float32).cpu()
    print(
        f"[{name}] shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
        f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare torch direct read with CIDS sync read.")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--page-size", type=int, default=131072)
    parser.add_argument("--cache-size", type=int, default=1024)
    parser.add_argument("--ctrl-idx", type=int, default=0)
    parser.add_argument("--start-sample-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--print-pixels", type=int, default=16)
    args = parser.parse_args()

    sample_ids = list(range(args.start_sample_id, args.start_sample_id + args.batch_size))
    print(f"[COMPARE] sample_ids={sample_ids}")

    torch_images, torch_labels = _load_torch_batch(args.prepared_root, sample_ids)
    print(f"[TORCH] labels={torch_labels.tolist()}")
    _print_tensor_stats("TORCH_IMAGES", torch_images)

    # CIDS 直读功能：
    # - 使用同一批 sample_id 走 sync 读取
    cids = CIDS.from_prepared_dataset(
        args.prepared_root,
        ctrl_idx=args.ctrl_idx,
        cache_size=args.cache_size,
        page_size=args.page_size,
    )
    cids_batch = cids.fetch_samples_sync(
        (
            torch.tensor(sample_ids, dtype=torch.long),
            torch_labels.clone(),
        ),
        device=f"cuda:{args.ctrl_idx}",
    )
    cids_images, cids_labels = _extract_images_and_labels(cids_batch)
    cids_images = cids_images.detach().cpu().to(torch.float32)
    cids_labels = cids_labels.detach().cpu().to(torch.long)

    print(f"[CIDS] labels={cids_labels.tolist()}")
    _print_tensor_stats("CIDS_IMAGES", cids_images)

    labels_equal = torch.equal(torch_labels.cpu(), cids_labels)
    diff = (torch_images.cpu() - cids_images).abs()
    print(f"[COMPARE] labels_equal={labels_equal}")
    print(
        f"[COMPARE] diff_max={diff.max().item():.8f} "
        f"diff_mean={diff.mean().item():.8f}"
    )

    per_sample_max = diff.view(diff.size(0), -1).max(dim=1).values
    per_sample_mean = diff.view(diff.size(0), -1).mean(dim=1)
    for idx, sample_id in enumerate(sample_ids):
        print(
            f"[SAMPLE] sample_id={sample_id} label={int(torch_labels[idx])} "
            f"max_abs_diff={per_sample_max[idx].item():.8f} "
            f"mean_abs_diff={per_sample_mean[idx].item():.8f}"
        )

    num_pixels = max(1, int(args.print_pixels))
    torch_prefix = torch_images[0].reshape(-1)[:num_pixels].tolist()
    cids_prefix = cids_images[0].reshape(-1)[:num_pixels].tolist()
    print(f"[PIXELS] torch_first_sample_prefix={torch_prefix}")
    print(f"[PIXELS] cids_first_sample_prefix={cids_prefix}")


if __name__ == "__main__":
    main()
