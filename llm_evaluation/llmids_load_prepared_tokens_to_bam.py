import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GIDS_SETUP_ROOT = REPO_ROOT / "GIDS_Setup"
if str(GIDS_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(GIDS_SETUP_ROOT))

from GIDS import LLMIDS


def _load_meta(prepared_root):
    with open(Path(prepared_root) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="把 prepared token dataset 写入 BaM")
    parser.add_argument("--train-root", required=True, help="prepared train dataset 目录")
    parser.add_argument("--val-root", default=None, help="prepared val dataset 目录，可选")
    parser.add_argument("--ctrl-idx", type=int, default=0, help="控制器/GPU 索引")
    parser.add_argument("--cache-size", type=int, default=10, help="BaM cache 大小，单位 MB")
    args = parser.parse_args()

    train_meta = _load_meta(args.train_root)
    llmids = LLMIDS.from_prepared_dataset(
        args.train_root,
        ctrl_idx=args.ctrl_idx,
        cache_size=args.cache_size,
    )

    train_info = llmids.load_prepared_tokens_to_bam(args.train_root, sample_id_offset=0)
    print(f"[LLMIDS_LOAD_TOKENS] train loaded: {train_info}", flush=True)

    if args.val_root is not None:
        val_info = llmids.load_prepared_tokens_to_bam(
            args.val_root,
            sample_id_offset=int(train_meta["num_samples"]),
        )
        print(f"[LLMIDS_LOAD_TOKENS] val loaded: {val_info}", flush=True)


if __name__ == "__main__":
    main()
