import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GIDS_SETUP_ROOT = REPO_ROOT / "GIDS_Setup"
if str(GIDS_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(GIDS_SETUP_ROOT))

from GIDS import CIDS


def _load_meta(prepared_root):
    # 读取 prepared dataset 的元信息，确定样本数和形状。
    meta_path = Path(prepared_root) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="把 prepared CIDS 数据集写入 BaM 数组")
    parser.add_argument("--train-root", required=True, help="prepared train dataset 目录")
    parser.add_argument("--val-root", default=None, help="prepared val dataset 目录，可选")
    parser.add_argument("--ctrl-idx", type=int, default=0, help="使用的 GPU/controller 索引")
    args = parser.parse_args()

    train_meta = _load_meta(args.train_root)
    cids = CIDS.from_prepared_dataset(
        args.train_root,
        ctrl_idx=args.ctrl_idx,
    )

    train_info = cids.load_prepared_images_to_bam(
        args.train_root,
        sample_id_offset=0,
    )
    print(
        f"[CIDS_LOAD] 训练集写入完成 root={train_info['prepared_root']} "
        f"num_samples={train_info['num_samples']} "
        f"sample_id_offset={train_info['sample_id_offset']}",
        flush=True,
    )

    # 先只验证 train 写入路径。
    # val 的带 offset 写入当前仍会触发 illegal memory access，暂时保留代码但不执行。
    #
    # if args.val_root is not None:
    #     val_start_sample_id = int(train_meta["num_samples"])
    #     val_info = cids.load_prepared_images_to_bam(
    #         args.val_root,
    #         sample_id_offset=val_start_sample_id,
    #     )
    #     print(
    #         f"[CIDS_LOAD] 验证集写入完成 root={val_info['prepared_root']} "
    #         f"num_samples={val_info['num_samples']} "
    #         f"sample_id_offset={val_info['sample_id_offset']}",
    #         flush=True,
    #     )


if __name__ == "__main__":
    main()
