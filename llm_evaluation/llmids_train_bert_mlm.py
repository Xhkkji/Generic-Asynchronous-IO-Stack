import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.profiler
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
GIDS_SETUP_ROOT = REPO_ROOT / "GIDS_Setup"
if str(GIDS_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(GIDS_SETUP_ROOT))

from GIDS import LLMIDS, LLMIDS_DataLoader, LLMIDSPreparedDataset


def _load_meta(prepared_root):
    with open(Path(prepared_root) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


class _TorchPreparedTokenDataset:
    # 用 torch 直接从 prepared token chunks 读取的最小 dataset。
    def __init__(self, prepared_root, torch_read_mode="mmap"):
        self.prepared_root = Path(prepared_root)
        self.meta = _load_meta(prepared_root)
        self.sample_dim = int(self.meta["sample_dim"])
        self.num_samples = int(self.meta["num_samples"])
        self.sequence_length = int(self.meta.get("sequence_length", self.sample_dim))
        self.pad_token_id = int(self.meta.get("pad_token_id", 0))
        self.lengths = np.load(self.prepared_root / self.meta.get("lengths_file", "lengths.npy"))
        tokens_path = self.prepared_root / self.meta.get("tokens_file", "tokens.bin")
        if torch_read_mode == "mmap":
            self.tokens = np.memmap(
                tokens_path,
                dtype=np.int64,
                mode="r",
                shape=(self.num_samples, self.sample_dim),
            )
        elif torch_read_mode == "buffered":
            self.tokens = np.fromfile(tokens_path, dtype=np.int64).reshape(
                self.num_samples, self.sample_dim
            )
        else:
            raise ValueError(f"不支持的 torch_read_mode: {torch_read_mode}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.from_numpy(np.array(self.tokens[idx], copy=True))
        length = int(self.lengths[idx])
        attention_mask = torch.zeros(self.sequence_length, dtype=torch.long)
        attention_mask[:length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def _build_torch_loader(prepared_root, batch_size, shuffle, torch_read_mode="mmap"):
    dataset = _TorchPreparedTokenDataset(prepared_root, torch_read_mode=torch_read_mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return dataset, dataloader


def _build_llmids_loader(prepared_root, batch_size, shuffle, llmids_loader, prefetch_depth,
                         start_sample_id=0, io_mode="registered", registered_split=1):
    dataset = LLMIDSPreparedDataset(prepared_root, start_sample_id=start_sample_id)
    dataloader = LLMIDS_DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        LLMIDS_Loader=llmids_loader,
        prefetch_depth=prefetch_depth,
        io_mode=io_mode,
        registered_split=registered_split,
    )
    return dataset, dataloader


def _apply_mlm_mask(input_ids, attention_mask, mask_token_id, pad_token_id, vocab_size, mlm_probability=0.15):
    labels = input_ids.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability, device=input_ids.device)
    probability_matrix.masked_fill_(attention_mask == 0, 0.0)
    if pad_token_id is not None:
        probability_matrix.masked_fill_(input_ids == pad_token_id, 0.0)

    masked_indices = torch.bernoulli(probability_matrix).to(torch.bool)
    labels.masked_fill_(~masked_indices, -100)

    replace_prob = torch.rand(labels.shape, device=input_ids.device)
    mask_replace = masked_indices & (replace_prob < 0.8)
    random_replace = masked_indices & (replace_prob >= 0.8) & (replace_prob < 0.9)

    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_replace] = mask_token_id
    if random_replace.any():
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
        masked_input_ids[random_replace] = random_words[random_replace]

    return masked_input_ids, labels


def _prepare_model_batch(batch, device, mask_token_id, pad_token_id, vocab_size):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if not torch.is_tensor(input_ids):
        input_ids = torch.as_tensor(input_ids)
    if not torch.is_tensor(attention_mask):
        attention_mask = torch.as_tensor(attention_mask)
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    masked_input_ids, labels = _apply_mlm_mask(
        input_ids,
        attention_mask,
        mask_token_id,
        pad_token_id,
        vocab_size,
    )
    return {
        "input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _start_profiler(enabled, output_dir):
    if not enabled:
        return None
    os.makedirs(output_dir, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    )
    profiler.__enter__()
    return profiler


def _stop_profiler(profiler):
    if profiler is None:
        return
    profiler.__exit__(None, None, None)
    try:
        key_averages = profiler.key_averages()
    except AssertionError:
        print("=== profiler has no finalized events to summarize ===", flush=True)
        return
    print("=== profiler summary: cuda_time_total ===", flush=True)
    print(key_averages.table(sort_by="cuda_time_total", row_limit=20), flush=True)
    print("=== profiler summary: cpu_time_total ===", flush=True)
    print(key_averages.table(sort_by="cpu_time_total", row_limit=100), flush=True)


def _masked_token_accuracy(logits, labels):
    mask = labels.ne(-100)
    if not mask.any():
        return 0, 0
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct, total


def train_one_epoch(model, dataloader, optimizer, device, mask_token_id, pad_token_id,
                    vocab_size, profiler=None, max_iters=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for step_idx, batch in enumerate(progress, start=1):
        model_batch = _prepare_model_batch(
            batch, device, mask_token_id, pad_token_id, vocab_size
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        correct, total = _masked_token_accuracy(outputs.logits, model_batch["labels"])
        total_loss += loss.item()
        total_correct += correct
        total_masked += total
        progress.set_postfix(
            loss=f"{(total_loss / max(1, step_idx)):.4f}",
            masked_acc=f"{(total_correct / max(1, total_masked)):.4f}",
        )
        if profiler is not None:
            profiler.step()
        if max_iters is not None and step_idx >= max_iters:
            break

    avg_loss = total_loss / max(1, step_idx)
    avg_acc = total_correct / max(1, total_masked)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataloader, device, mask_token_id, pad_token_id, vocab_size):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    progress = tqdm(dataloader, desc="val", leave=False)
    for step_idx, batch in enumerate(progress, start=1):
        model_batch = _prepare_model_batch(
            batch, device, mask_token_id, pad_token_id, vocab_size
        )
        outputs = model(**model_batch)
        correct, total = _masked_token_accuracy(outputs.logits, model_batch["labels"])
        total_loss += outputs.loss.item()
        total_correct += correct
        total_masked += total
        progress.set_postfix(
            loss=f"{(total_loss / max(1, step_idx)):.4f}",
            masked_acc=f"{(total_correct / max(1, total_masked)):.4f}",
        )

    avg_loss = total_loss / max(1, step_idx)
    avg_acc = total_correct / max(1, total_masked)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="最小 LLMIDS + BERT MLM baseline")
    parser.add_argument("--train-root", required=True, help="prepared train dataset 目录")
    parser.add_argument("--val-root", default=None, help="prepared val dataset 目录，可选")
    parser.add_argument("--model-name-or-path", default="bert-base-chinese", help="BERT 模型名或本地路径")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--max-train-iters", type=int, default=0, help="每个 epoch 最多跑多少个 iter；0 表示不限制")
    parser.add_argument("--run-val", choices=["0", "1"], default="1", help="是否运行验证")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--prefetch-depth", type=int, default=4, help="LLMIDS 预取深度")
    parser.add_argument("--cache-size", type=int, default=10, help="LLMIDS/BaM cache 大小，单位 MB")
    parser.add_argument("--registered-split", type=int, default=1, help="registered 模式下一个 batch 拆成多少个 sub-request")
    parser.add_argument("--ctrl-idx", type=int, default=0, help="使用哪块 GPU/控制器")
    parser.add_argument("--io-mode", choices=["torch", "sync", "registered"], default="torch", help="读取路径")
    parser.add_argument("--torch-read-mode", choices=["mmap", "buffered"], default="mmap", help="torch 模式下读取方式")
    parser.add_argument("--enable-profile", choices=["0", "1"], default="0", help="是否开启 profiler")
    parser.add_argument("--profile-dir", default="./llmids_profile", help="profiler 输出目录")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("运行 BERT MLM baseline 需要 transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    if tokenizer.mask_token_id is None:
        raise ValueError("tokenizer 没有 mask_token_id，无法运行 MLM")

    run_val = args.run_val == "1" and args.val_root is not None
    max_train_iters = args.max_train_iters if args.max_train_iters > 0 else None
    enable_profile = args.enable_profile == "1"

    if args.io_mode == "registered":
        os.environ.setdefault("GIDS_FORCE_SYNC_READ", "0")
    elif args.io_mode == "sync":
        os.environ["GIDS_FORCE_SYNC_READ"] = "1"
    else:
        os.environ.setdefault("GIDS_FORCE_SYNC_READ", "0")
    os.environ["CIDS_REGISTERED_SPLIT"] = str(max(1, args.registered_split))

    device = torch.device(f"cuda:{args.ctrl_idx}" if torch.cuda.is_available() else "cpu")
    train_meta = _load_meta(args.train_root)

    if args.io_mode == "torch":
        _, train_loader = _build_torch_loader(
            prepared_root=args.train_root,
            batch_size=args.batch_size,
            shuffle=True,
            torch_read_mode=args.torch_read_mode,
        )
        val_loader = None
        if run_val:
            _, val_loader = _build_torch_loader(
                prepared_root=args.val_root,
                batch_size=args.batch_size,
                shuffle=False,
                torch_read_mode=args.torch_read_mode,
            )
    else:
        train_llmids = LLMIDS.from_prepared_dataset(
            args.train_root,
            ctrl_idx=args.ctrl_idx,
            cache_size=args.cache_size,
        )
        _, train_loader = _build_llmids_loader(
            prepared_root=args.train_root,
            batch_size=args.batch_size,
            shuffle=True,
            llmids_loader=train_llmids,
            prefetch_depth=args.prefetch_depth,
            start_sample_id=0,
            io_mode=args.io_mode,
            registered_split=args.registered_split,
        )
        val_loader = None
        if run_val:
            _, val_loader = _build_llmids_loader(
                prepared_root=args.val_root,
                batch_size=args.batch_size,
                shuffle=False,
                llmids_loader=train_llmids,
                prefetch_depth=args.prefetch_depth,
                start_sample_id=int(train_meta["num_samples"]),
                io_mode=args.io_mode,
                registered_split=args.registered_split,
            )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(
        f"[LLMIDS_BERT_MLM] device={device} io_mode={args.io_mode} "
        f"torch_read_mode={args.torch_read_mode} batch_size={args.batch_size} "
        f"max_train_iters={max_train_iters if max_train_iters is not None else 'full'} "
        f"sequence_length={train_meta.get('sequence_length', train_meta['shape'][0])}",
        flush=True,
    )

    profiler = _start_profiler(enable_profile, args.profile_dir)
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                tokenizer.mask_token_id,
                tokenizer.pad_token_id,
                tokenizer.vocab_size,
                profiler=profiler,
                max_iters=max_train_iters,
            )
            print(
                f"[LLMIDS_BERT_MLM] epoch={epoch} train_loss={train_loss:.4f} "
                f"train_masked_acc={train_acc:.4f}",
                flush=True,
            )
            if run_val and val_loader is not None:
                val_loss, val_acc = evaluate(
                    model,
                    val_loader,
                    device,
                    tokenizer.mask_token_id,
                    tokenizer.pad_token_id,
                    tokenizer.vocab_size,
                )
                print(
                    f"[LLMIDS_BERT_MLM] epoch={epoch} val_loss={val_loss:.4f} "
                    f"val_masked_acc={val_acc:.4f}",
                    flush=True,
                )
    finally:
        _stop_profiler(profiler)


if __name__ == "__main__":
    main()
