import argparse
import json
import random
import shutil
from pathlib import Path


def _list_class_dirs(split_root: Path):
    return sorted([p for p in split_root.iterdir() if p.is_dir()])


def _list_images(class_root: Path):
    return sorted([p for p in class_root.iterdir() if p.is_file()])


def _sample_paths(paths, keep_count, rng):
    if len(paths) < keep_count:
        raise ValueError(
            f"class {paths[0].parent.name if paths else 'UNKNOWN'} only has {len(paths)} images, "
            f"cannot keep {keep_count}"
        )
    indices = sorted(rng.sample(range(len(paths)), keep_count))
    return [paths[idx] for idx in indices]


def _sample_disjoint_train_val(paths, train_count, val_count, rng):
    total_count = train_count + val_count
    if len(paths) < total_count:
        raise ValueError(
            f"class {paths[0].parent.name if paths else 'UNKNOWN'} only has {len(paths)} images, "
            f"cannot keep train={train_count} val={val_count}"
        )
    indices = sorted(rng.sample(range(len(paths)), total_count))
    selected = [paths[idx] for idx in indices]
    return selected[:train_count], selected[train_count:]


def _symlink_subset(src_split_root: Path, dst_split_root: Path, keep_count: int, seed: int):
    rng = random.Random(seed)
    class_dirs = _list_class_dirs(src_split_root)
    dst_split_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for class_dir in class_dirs:
        images = _list_images(class_dir)
        selected = _sample_paths(images, keep_count, rng)
        dst_class_dir = dst_split_root / class_dir.name
        if dst_class_dir.exists():
            shutil.rmtree(dst_class_dir)
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for src_path in selected:
            dst_path = dst_class_dir / src_path.name
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            dst_path.symlink_to(src_path)

        manifest.append(
            {
                "class_name": class_dir.name,
                "selected_count": len(selected),
            }
        )

    return manifest


def _symlink_disjoint_train_val(
    src_train_root: Path,
    dst_train_root: Path,
    dst_val_root: Path,
    train_count: int,
    val_count: int,
    seed: int,
):
    rng = random.Random(seed)
    class_dirs = _list_class_dirs(src_train_root)
    dst_train_root.mkdir(parents=True, exist_ok=True)
    dst_val_root.mkdir(parents=True, exist_ok=True)

    train_manifest = []
    val_manifest = []
    for class_dir in class_dirs:
        images = _list_images(class_dir)
        selected_train, selected_val = _sample_disjoint_train_val(images, train_count, val_count, rng)

        dst_train_class_dir = dst_train_root / class_dir.name
        dst_val_class_dir = dst_val_root / class_dir.name
        if dst_train_class_dir.exists():
            shutil.rmtree(dst_train_class_dir)
        if dst_val_class_dir.exists():
            shutil.rmtree(dst_val_class_dir)
        dst_train_class_dir.mkdir(parents=True, exist_ok=True)
        dst_val_class_dir.mkdir(parents=True, exist_ok=True)

        for src_path in selected_train:
            (dst_train_class_dir / src_path.name).symlink_to(src_path)
        for src_path in selected_val:
            (dst_val_class_dir / src_path.name).symlink_to(src_path)

        train_manifest.append({"class_name": class_dir.name, "selected_count": len(selected_train)})
        val_manifest.append({"class_name": class_dir.name, "selected_count": len(selected_val)})

    return train_manifest, val_manifest


def build_subset(
    input_root: Path,
    output_root: Path,
    train_per_class: int,
    val_per_class: int,
    seed: int,
):
    train_src = input_root / "train"
    if not train_src.is_dir():
        raise FileNotFoundError(f"expected ImageFolder train under {input_root}")

    train_dst = output_root / "train"
    val_dst = output_root / "val"
    train_manifest, val_manifest = _symlink_disjoint_train_val(
        train_src,
        train_dst,
        val_dst,
        train_per_class,
        val_per_class,
        seed,
    )

    meta = {
        "source_root": str(input_root),
        "output_root": str(output_root),
        "subset_strategy": "train_only_disjoint_split",
        "train_per_class": int(train_per_class),
        "val_per_class": int(val_per_class),
        "seed": int(seed),
        "num_classes": len(train_manifest),
        "train_total": len(train_manifest) * int(train_per_class),
        "val_total": len(val_manifest) * int(val_per_class),
    }
    with open(output_root / "subset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        "[IMAGENET_SUBSET] done "
        f"num_classes={meta['num_classes']} "
        f"train_total={meta['train_total']} "
        f"val_total={meta['val_total']} "
        f"output_root={output_root}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Build balanced ImageNet-1K subset using symlinks.")
    parser.add_argument(
        "--input-root",
        default="/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_rgb",
        help="source ImageFolder root",
    )
    parser.add_argument(
        "--output-root",
        default="/home/xhk/hyperion/GIDS/dataset/imagenet/imagenet1k_subset_190_20_pad448/imagenet1k_rgb_subset_190_20",
        help="subset ImageFolder root",
    )
    parser.add_argument("--train-per-class", type=int, default=190)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    build_subset(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
