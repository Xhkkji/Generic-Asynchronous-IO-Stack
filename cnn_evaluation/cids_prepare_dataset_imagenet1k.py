import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


def _load_meta(root: Path):
    with open(root / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _dtype_name_to_numpy(dtype_name: str):
    dtype_map = {
        "uint8": np.uint8,
        "float16": np.float16,
        "float32": np.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"不支持的 dtype: {dtype_name}")
    return dtype_map[dtype_name]


def _materialize_bam_aligned_dataset(src_root: Path, dst_root: Path, page_size: int, chunk_samples: int):
    meta = _load_meta(src_root)
    num_samples = int(meta["num_samples"])
    sample_dim = int(meta["sample_dim"])
    dtype_name = str(meta["dtype"])
    np_dtype = _dtype_name_to_numpy(dtype_name)
    itemsize = np.dtype(np_dtype).itemsize

    sample_bytes = sample_dim * itemsize
    padded_sample_bytes = math.ceil(sample_bytes / page_size) * page_size
    padded_sample_dim = padded_sample_bytes // itemsize

    dst_root.mkdir(parents=True, exist_ok=True)

    src_images = src_root / meta.get("images_file", "images.bin")
    dst_images = dst_root / meta.get("images_file", "images.bin")
    src_labels = src_root / meta.get("labels_file", "labels.npy")
    dst_labels = dst_root / meta.get("labels_file", "labels.npy")
    dst_meta = dst_root / "meta.json"

    src_arr = np.memmap(src_images, dtype=np_dtype, mode="r", shape=(num_samples, sample_dim))
    dst_arr = np.memmap(dst_images, dtype=np_dtype, mode="w+", shape=(num_samples, padded_sample_dim))
    total_chunks = math.ceil(num_samples / chunk_samples)
    for chunk_idx, start in enumerate(range(0, num_samples, chunk_samples), start=1):
        end = min(start + chunk_samples, num_samples)
        dst_arr[start:end] = 0
        dst_arr[start:end, :sample_dim] = src_arr[start:end]
        dst_arr.flush()
        print(
            f"[CIDS_PREP_IMAGENET1K] align {src_root.name} "
            f"chunk={chunk_idx}/{total_chunks} samples={end}/{num_samples}",
            flush=True,
        )
    dst_arr.flush()

    shutil.copy2(src_labels, dst_labels)

    out_meta = dict(meta)
    out_meta["sample_bytes"] = padded_sample_bytes
    with open(dst_meta, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)

    return {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "dtype": dtype_name,
        "num_samples": num_samples,
        "sample_dim": sample_dim,
        "sample_bytes": sample_bytes,
        "padded_sample_dim": padded_sample_dim,
        "padded_sample_bytes": padded_sample_bytes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="为 ImageNet-1K prepared dataset 生成适合 BaM 固定页布局写盘的对齐版本"
    )
    parser.add_argument(
        "--train-root",
        default="/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8",
        help="原始 prepared train 目录",
    )
    parser.add_argument(
        "--val-root",
        default="/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8",
        help="原始 prepared val 目录",
    )
    parser.add_argument(
        "--train-bam-root",
        default="/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_train_u8_bam",
        help="输出的 BaM 对齐版 train 目录",
    )
    parser.add_argument(
        "--val-bam-root",
        default="/home/xhk/hyperion/GIDS/dataset/imagenet/cids_imagenet1k_val_u8_bam",
        help="输出的 BaM 对齐版 val 目录",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=4096,
        help="BaM 页大小，默认 4096",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=4096,
        help="一次补齐和刷盘多少个样本，默认 4096",
    )
    args = parser.parse_args()

    print("[CIDS_PREP_IMAGENET1K] 开始对齐 train dataset", flush=True)
    train_info = _materialize_bam_aligned_dataset(
        Path(args.train_root),
        Path(args.train_bam_root),
        args.page_size,
        args.chunk_samples,
    )
    print("[CIDS_PREP_IMAGENET1K] 开始对齐 val dataset", flush=True)
    val_info = _materialize_bam_aligned_dataset(
        Path(args.val_root),
        Path(args.val_bam_root),
        args.page_size,
        args.chunk_samples,
    )

    print(
        "[CIDS_PREP_IMAGENET1K] train aligned "
        f"sample_bytes={train_info['sample_bytes']} -> {train_info['padded_sample_bytes']} "
        f"dst={train_info['dst_root']}",
        flush=True,
    )
    print(
        "[CIDS_PREP_IMAGENET1K] val aligned "
        f"sample_bytes={val_info['sample_bytes']} -> {val_info['padded_sample_bytes']} "
        f"dst={val_info['dst_root']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
