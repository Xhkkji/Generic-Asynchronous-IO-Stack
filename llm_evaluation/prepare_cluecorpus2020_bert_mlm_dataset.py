import argparse
import json
from pathlib import Path

import numpy as np


TEXT_KEYS = ("text", "content", "sentence", "body", "title")


def _iter_texts(input_root):
    for path in sorted(Path(input_root).rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        yield text
            continue

        if suffix in (".jsonl", ".json"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, str):
                        text = obj.strip()
                        if text:
                            yield text
                        continue
                    if not isinstance(obj, dict):
                        continue
                    for key in TEXT_KEYS:
                        value = obj.get(key)
                        if isinstance(value, str) and value.strip():
                            yield value.strip()
                            break


def _tokenize_to_chunks(text_iter, tokenizer, seq_len):
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        raise ValueError("tokenizer 需要同时提供 cls/sep/pad token")

    usable_len = seq_len - 2
    chunks = []
    lengths = []

    for text in text_iter:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        for start in range(0, len(token_ids), usable_len):
            piece = token_ids[start:start + usable_len]
            sample = [cls_id] + piece + [sep_id]
            length = len(sample)
            if length < seq_len:
                sample.extend([pad_id] * (seq_len - length))
            chunks.append(sample)
            lengths.append(length)

    if not chunks:
        raise ValueError("没有从输入目录中解析出任何文本 chunk")

    return np.asarray(chunks, dtype=np.int64), np.asarray(lengths, dtype=np.int64)


def _write_split(output_root, tokens, lengths, seq_len, tokenizer):
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tokens.bin").write_bytes(tokens.tobytes())
    np.save(output_root / "lengths.npy", lengths)
    meta = {
        "dataset": "cluecorpus2020",
        "task": "masked_language_modeling",
        "num_samples": int(tokens.shape[0]),
        "shape": [int(seq_len)],
        "sequence_length": int(seq_len),
        "dtype": "int64",
        "sample_dim": int(seq_len),
        "sample_bytes": int(seq_len * 8),
        "tokens_file": "tokens.bin",
        "lengths_file": "lengths.npy",
        "pad_token_id": int(tokenizer.pad_token_id),
        "vocab_size": int(tokenizer.vocab_size),
        "return_labels": False,
    }
    with open(output_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="把 CLUECorpus2020 预处理成 LLMIDS 可读的 BERT MLM fixed-length token chunks")
    parser.add_argument("--input-root", required=True, help="CLUECorpus2020 原始文本目录")
    parser.add_argument("--output-root", required=True, help="输出根目录，例如 dataset/cluecorpus2020/bert_base_seq256")
    parser.add_argument("--model-name-or-path", default="bert-base-chinese", help="Tokenizer 对应模型名或本地路径")
    parser.add_argument("--seq-len", type=int, default=256, help="固定 chunk 长度")
    parser.add_argument("--validation-ratio", type=float, default=0.01, help="验证集比例")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("准备 CLUECorpus2020 需要 transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokens, lengths = _tokenize_to_chunks(
        _iter_texts(args.input_root),
        tokenizer,
        args.seq_len,
    )

    num_samples = tokens.shape[0]
    num_val = max(1, int(num_samples * args.validation_ratio))
    num_train = max(1, num_samples - num_val)
    train_tokens = tokens[:num_train]
    train_lengths = lengths[:num_train]
    val_tokens = tokens[num_train:]
    val_lengths = lengths[num_train:]

    output_root = Path(args.output_root)
    _write_split(output_root / "train", train_tokens, train_lengths, args.seq_len, tokenizer)
    _write_split(output_root / "val", val_tokens, val_lengths, args.seq_len, tokenizer)

    print(
        f"[PREP_CLUE_BERT_MLM] num_train={len(train_tokens)} num_val={len(val_tokens)} "
        f"seq_len={args.seq_len} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
