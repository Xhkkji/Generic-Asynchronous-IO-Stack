import argparse
import os
import shutil
from pathlib import Path


def _load_imagenet1k_class_order(devkit_root: Path):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("需要 scipy 来解析 ImageNet devkit 的 meta.mat") from exc

    meta_path = devkit_root / "data" / "meta.mat"
    gt_path = devkit_root / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not meta_path.exists():
        raise FileNotFoundError(f"未找到 meta.mat: {meta_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"未找到验证集标签文件: {gt_path}")

    meta = loadmat(meta_path, squeeze_me=True)
    synsets = meta["synsets"]

    # 只保留叶子类，并按 ILSVRC2012_ID 排序，得到官方 1000 类顺序。
    leaf_entries = []
    for syn in synsets:
        ilsvrc_id = int(syn[0])
        wnid = str(syn[1])
        num_children = int(syn[4])
        if num_children == 0:
            leaf_entries.append((ilsvrc_id, wnid))
    leaf_entries.sort(key=lambda item: item[0])

    class_ids = [class_id for class_id, _ in leaf_entries]
    wnids = [wnid for _, wnid in leaf_entries]
    if len(wnids) != 1000:
        raise ValueError(f"解析得到的叶子类数量不是 1000，而是 {len(wnids)}")

    id_to_wnid = dict(zip(class_ids, wnids))

    with gt_path.open("r", encoding="utf-8") as f:
        val_gt = [int(line.strip()) for line in f if line.strip()]

    return id_to_wnid, val_gt


def _ensure_train_link(output_root: Path, train_raw: Path | None):
    if train_raw is None:
        print("[IMAGENET1K_LAYOUT] 未找到 train_raw/train，先跳过 train 布局", flush=True)
        return
    train_link = output_root / "train"
    if train_link.is_symlink() or train_link.exists():
        return
    os.symlink(train_raw, train_link)


def _build_val_layout(output_root: Path, val_raw: Path, id_to_wnid, val_gt):
    val_out = output_root / "val"
    if val_out.exists() or val_out.is_symlink():
        shutil.rmtree(val_out)
    val_out.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in val_raw.iterdir() if path.is_file())
    if len(images) != len(val_gt):
        raise ValueError(
            f"验证集图片数量 {len(images)} 与 ground truth 数量 {len(val_gt)} 不一致"
        )

    for idx, (image_path, class_id) in enumerate(zip(images, val_gt), start=1):
        wnid = id_to_wnid[class_id]
        class_dir = val_out / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        dst = class_dir / image_path.name
        if not dst.exists():
            os.symlink(image_path, dst)
        if idx % 1000 == 0:
            print(
                f"[IMAGENET1K_LAYOUT] val {idx}/{len(images)} -> {wnid}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="把解压后的 ImageNet-1K 整理成 ImageFolder 风格目录"
    )
    parser.add_argument("--imagenet-root", required=True, help="Imagenet2012 根目录")
    parser.add_argument(
        "--output-root",
        required=True,
        help="输出的 ImageFolder 根目录，例如 imagenet1k_rgb",
    )
    args = parser.parse_args()

    imagenet_root = Path(args.imagenet_root).resolve()
    output_root = Path(args.output_root).resolve()

    train_raw = imagenet_root / "train_raw"
    if not train_raw.is_dir():
        train_raw = imagenet_root / "train"
    if not train_raw.is_dir():
        train_raw = None

    val_raw = imagenet_root / "val_raw"
    if not val_raw.is_dir():
        val_raw = imagenet_root / "val"
    devkit_root = imagenet_root / "ILSVRC2012_devkit_t12"

    if not val_raw.is_dir():
        raise FileNotFoundError(
            f"未找到 val_raw/val 目录: {imagenet_root / 'val_raw'} 或 {imagenet_root / 'val'}"
        )
    if not devkit_root.is_dir():
        raise FileNotFoundError(f"未找到 devkit 目录: {devkit_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    id_to_wnid, val_gt = _load_imagenet1k_class_order(devkit_root)

    _ensure_train_link(output_root, train_raw)
    _build_val_layout(output_root, val_raw, id_to_wnid, val_gt)

    print(f"[IMAGENET1K_LAYOUT] 完成 output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
