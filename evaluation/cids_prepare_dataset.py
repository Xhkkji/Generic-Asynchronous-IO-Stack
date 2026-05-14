import argparse
import json
import os

import numpy as np
from PIL import Image


def _ensure_rgb(image):
    # 统一转成 RGB，避免灰度图或带 alpha 图导致 shape 不一致。
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _build_tiny_imagenet_samples(root, split):
    # 收集 Tiny ImageNet 的 (image_path, label) 列表。
    wnids_path = os.path.join(root, "wnids.txt")
    if not os.path.exists(wnids_path):
        raise FileNotFoundError(f"未找到 Tiny ImageNet 的 wnids.txt: {wnids_path}")

    with open(wnids_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    samples = []
    if split == "train":
        train_root = os.path.join(root, "train")
        for cls_name in classes:
            img_dir = os.path.join(train_root, cls_name, "images")
            if not os.path.isdir(img_dir):
                continue
            image_names = sorted(
                name for name in os.listdir(img_dir)
                if not name.startswith("."))
            for image_name in image_names:
                samples.append((os.path.join(img_dir, image_name), class_to_idx[cls_name]))
    else:
        val_root = os.path.join(root, "val")
        anno_path = os.path.join(val_root, "val_annotations.txt")
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"未找到 Tiny ImageNet 验证集标注文件: {anno_path}")
        images_dir = os.path.join(val_root, "images")
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, cls_name = parts[0], parts[1]
                if cls_name not in class_to_idx:
                    continue
                samples.append((os.path.join(images_dir, image_name), class_to_idx[cls_name]))

    return samples, classes, class_to_idx


def _build_imagenet1k_samples(root, split):
    # 使用 ImageFolder 风格目录收集 ImageNet-1K 的样本路径。
    try:
        from torchvision.datasets import ImageFolder
    except ImportError as exc:
        raise ImportError("准备 ImageNet-1K 需要 torchvision") from exc

    split_root = os.path.join(root, split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"未找到 ImageNet-1K {split} 目录: {split_root}")

    dataset = ImageFolder(split_root)
    samples = [(path, label) for path, label in dataset.samples]
    return samples, dataset.classes, dataset.class_to_idx


def _pil_to_chw_float32(image):
    # 把 PIL 图片转成 CHW float32 tensor 语义的 numpy 数组，范围 0~1。
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def _resize_to_canvas_with_padding(image, target_hw):
    # 尽量保留原图信息：
    # - 保持长宽比
    # - 仅在原图超过目标画布时做等比例缩小
    # - 最后居中 pad 到固定大小
    target_h, target_w = target_hw
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"非法图像尺寸: {(src_w, src_h)}")

    scale = min(target_w / src_w, target_h / src_h, 1.0)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))

    if resized_w != src_w or resized_h != src_h:
        image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    canvas.paste(image, (pad_left, pad_top))
    return canvas


def _build_transform(dataset_name, preprocess_mode="legacy"):
    # 构建固定尺寸的确定性预处理。
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("准备 CIDS 数据集需要 torchvision") from exc

    if dataset_name == "tiny-imagenet":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]), [3, 64, 64]

    if dataset_name == "imagenet1k":
        if preprocess_mode == "legacy":
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]), [3, 224, 224]
        if preprocess_mode == "pad448":
            def _transform(image):
                padded = _resize_to_canvas_with_padding(image, (448, 448))
                return _pil_to_chw_float32(padded)

            return _transform, [3, 448, 448]
        raise ValueError(f"imagenet1k 不支持的 preprocess_mode: {preprocess_mode}")

    raise ValueError(f"不支持的数据集类型: {dataset_name}")


def _tensor_to_storage_array(tensor, dtype_name):
    # 把 CHW tensor 转成最终写盘格式。
    tensor = np.asarray(tensor)
    if dtype_name == "float16":
        return tensor.astype(np.float16, copy=False)
    if dtype_name == "float32":
        return tensor.astype(np.float32, copy=False)
    if dtype_name == "uint8":
        array = np.clip(np.rint(tensor * 255.0), 0, 255)
        return array.astype(np.uint8)
    raise ValueError(f"不支持的 dtype: {dtype_name}")


def prepare_cids_dataset(
    dataset_name,
    input_root,
    output_root,
    split,
    dtype_name,
    preprocess_mode="legacy",
):
    # 把原始图片整理成固定尺寸 tensor，并按 sample_id 顺序线性写盘。
    dataset_name = dataset_name.lower()
    split = split.lower()
    os.makedirs(output_root, exist_ok=True)

    if dataset_name == "tiny-imagenet":
        samples, classes, class_to_idx = _build_tiny_imagenet_samples(input_root, split)
    elif dataset_name == "imagenet1k":
        samples, classes, class_to_idx = _build_imagenet1k_samples(input_root, split)
    else:
        raise ValueError("dataset_name 只支持 tiny-imagenet 或 imagenet1k")

    transform, sample_shape = _build_transform(dataset_name, preprocess_mode=preprocess_mode)
    images_path = os.path.join(output_root, "images.bin")
    labels_path = os.path.join(output_root, "labels.npy")
    meta_path = os.path.join(output_root, "meta.json")

    labels = np.empty(len(samples), dtype=np.int64)

    with open(images_path, "wb") as f:
        for sample_id, (image_path, label) in enumerate(samples):
            with Image.open(image_path) as image:
                image = _ensure_rgb(image)
                transformed = transform(image)
                if hasattr(transformed, "numpy"):
                    tensor = transformed.numpy()
                else:
                    tensor = np.asarray(transformed, dtype=np.float32)
            array = _tensor_to_storage_array(tensor, dtype_name)
            array.tofile(f)
            labels[sample_id] = label

            if sample_id % 1000 == 0:
                print(
                    f"[CIDS_PREP] split={split} sample_id={sample_id}/{len(samples)} "
                    f"path={image_path}",
                    flush=True,
                )

    np.save(labels_path, labels)

    dtype_to_itemsize = {
        "uint8": 1,
        "float16": 2,
        "float32": 4,
    }
    sample_dim = int(np.prod(sample_shape))
    sample_bytes = sample_dim * dtype_to_itemsize[dtype_name]

    meta = {
        "dataset": dataset_name,
        "split": split,
        "num_samples": int(len(samples)),
        "shape": sample_shape,
        "dtype": dtype_name,
        "preprocess_mode": preprocess_mode,
        "sample_dim": sample_dim,
        "sample_bytes": sample_bytes,
        "images_file": "images.bin",
        "labels_file": "labels.npy",
        "classes": classes,
        "class_to_idx": class_to_idx,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[CIDS_PREP] 完成 dataset={dataset_name}, split={split}, "
        f"num_samples={len(samples)}, shape={sample_shape}, dtype={dtype_name}, "
        f"preprocess_mode={preprocess_mode}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="为 CIDS 准备固定尺寸图片 tensor 数据集")
    parser.add_argument(
        "--dataset",
        choices=["tiny-imagenet", "imagenet1k"],
        required=True,
        help="数据集类型",
    )
    parser.add_argument("--input-root", required=True, help="原始数据集根目录")
    parser.add_argument("--output-root", required=True, help="整理后输出目录")
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="数据集划分",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "uint8"],
        default="float16",
        help="写盘数据类型，默认 float16",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["legacy", "pad448"],
        default="legacy",
        help="图片预处理分支：legacy 为原来的 Resize256+CenterCrop224；pad448 为等比例缩小到 448 画布内后居中 padding",
    )
    args = parser.parse_args()

    prepare_cids_dataset(
        dataset_name=args.dataset,
        input_root=args.input_root,
        output_root=args.output_root,
        split=args.split,
        dtype_name=args.dtype,
        preprocess_mode=args.preprocess_mode,
    )


if __name__ == "__main__":
    main()
