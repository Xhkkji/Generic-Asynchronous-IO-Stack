import json
from pathlib import Path


def _load_meta(prepared_root):
    meta_path = Path(prepared_root) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 这个脚本现在只保留为元信息说明入口。
    # CIDS 的正式写盘路径已经切换到 BaM 自带的
    # nvm-readwrite_stripe-bench，请使用 cids_load_to_bam.sh。
    train_root = Path("/home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_train_f32")
    val_root = Path("/home/xhk/hyperion/GIDS/dataset/imagenet/cids_tiny_imagenet_val_f32")

    train_meta = _load_meta(train_root)
    val_meta = _load_meta(val_root)
    train_bytes = int(train_meta["num_samples"]) * int(train_meta["sample_bytes"])

    print("[CIDS_LOAD] 当前推荐使用 BaM benchmark 写盘，而不是 Python store_tensor 路径。", flush=True)
    print(f"[CIDS_LOAD] train_root={train_root}", flush=True)
    print(f"[CIDS_LOAD] val_root={val_root}", flush=True)
    print(f"[CIDS_LOAD] train_bytes={train_bytes}", flush=True)
    print(f"[CIDS_LOAD] val_loffset_bytes={train_bytes}", flush=True)
    print("[CIDS_LOAD] 请直接运行: bash /home/xhk/hyperion/GIDS/cnn_evaluation/common_tools/cids_load_to_bam.sh", flush=True)


if __name__ == "__main__":
    main()
