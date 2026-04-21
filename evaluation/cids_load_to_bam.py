import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GIDS_SETUP_ROOT = REPO_ROOT / "GIDS_Setup"
if str(GIDS_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(GIDS_SETUP_ROOT))

from GIDS import CIDS


def main():
    parser = argparse.ArgumentParser(description="把 prepared CIDS 数据集写入 BaM 数组")
    parser.add_argument("--train-root", required=True, help="prepared train dataset 目录")
    parser.add_argument("--val-root", default=None, help="prepared val dataset 目录，可选")
    parser.add_argument("--ctrl-idx", type=int, default=0, help="使用的 GPU/controller 索引")
    args = parser.parse_args()

    cids = CIDS.from_prepared_dataset(
        args.train_root,
        ctrl_idx=args.ctrl_idx,
    )

    prepared_roots = [args.train_root]
    if args.val_root is not None:
        prepared_roots.append(args.val_root)

    infos = cids.load_combined_prepared_images_to_bam(prepared_roots)
    for info in infos:
        split_name = "训练集" if Path(info["prepared_root"]).resolve() == Path(args.train_root).resolve() else "验证集"
        print(
            f"[CIDS_LOAD] {split_name}写入完成 root={info['prepared_root']} "
            f"num_samples={info['num_samples']} "
            f"sample_id_offset={info['sample_id_offset']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
