from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BAMPolicyStats:
    # BaM policy 统计：
    # - update_calls: 做了多少次 batch 内评分更新
    # - sync_calls: 向 BaM 同步了多少次整表分数
    # - tracked_pages: 当前 GPU policy table 覆盖多少个 policy slot（通常等于样本数）
    # - last_batch_pages: 最近一个 batch 实际更新了多少个 policy slot
    update_calls: int = 0
    sync_calls: int = 0
    tracked_pages: int = 0
    last_batch_pages: int = 0


class GPUBAMPolicy:
    # GPU 版 BaM policy：
    # - 功能：在 GPU 上维护 per-sample importance 分数
    # - 思路：batch loss -> sample score -> GPU EMA table -> 整表 D2D 同步给 BaM
    # - 当 1 sample 对应多个 row/page 时，BaM 侧用 grouped policy 接口把这些 row 映射到同一条分数
    def __init__(self, num_pages, device, ema_alpha=0.9):
        num_pages = int(num_pages)
        if num_pages <= 0:
            raise ValueError("GPUBAMPolicy 需要正数 num_pages")

        ema_alpha = float(ema_alpha)
        if not 0.0 <= ema_alpha < 1.0:
            raise ValueError("ema_alpha 必须满足 0 <= ema_alpha < 1")

        self.device = torch.device(device)
        self.ema_alpha = ema_alpha
        self.policy_scores = torch.zeros(num_pages, dtype=torch.float32, device=self.device)
        self.seen_counts = torch.zeros(num_pages, dtype=torch.int32, device=self.device)
        self.stats = BAMPolicyStats(tracked_pages=num_pages)

    def update_from_batch(self, sample_ids, sample_scores):
        # 训练阶段功能：
        # - 输入当前 batch 的 sample_ids / sample_scores
        # - 在 GPU 上更新对应页的 EMA importance
        if sample_ids is None or sample_scores is None:
            self.stats.last_batch_pages = 0
            return 0

        sample_ids = sample_ids.detach().to(device=self.device, dtype=torch.long).view(-1)
        sample_scores = sample_scores.detach().to(device=self.device, dtype=torch.float32).view(-1)
        if sample_ids.numel() == 0:
            self.stats.last_batch_pages = 0
            return 0
        if sample_ids.numel() != sample_scores.numel():
            raise ValueError("update_from_batch 需要等长的 sample_ids 和 sample_scores")

        # batch 内按 sample 去重，保留最后一次分数，代码尽量简单。
        unique_ids, inverse = torch.unique(sample_ids, sorted=False, return_inverse=True)
        batch_scores = torch.zeros(unique_ids.shape[0], dtype=torch.float32, device=self.device)
        batch_scores[inverse] = sample_scores

        prev_scores = self.policy_scores[unique_ids]
        prev_seen = self.seen_counts[unique_ids] > 0
        updated_scores = torch.where(
            prev_seen,
            self.ema_alpha * prev_scores + (1.0 - self.ema_alpha) * batch_scores,
            batch_scores,
        )
        self.policy_scores[unique_ids] = updated_scores
        self.seen_counts[unique_ids] = self.seen_counts[unique_ids] + 1

        updated = int(unique_ids.numel())
        self.stats.update_calls += 1
        self.stats.last_batch_pages = updated
        return updated

    def sync_to_bam(self, cids_loader):
        # 同步功能：
        # - 把整张 GPU policy table 直接 D2D 拷贝到 BaM 的 policy table
        if cids_loader is None:
            return
        cids_loader.sync_bam_policy_scores_device(self.policy_scores)
        self.stats.sync_calls += 1

    def summary(self):
        return {
            "update_calls": int(self.stats.update_calls),
            "sync_calls": int(self.stats.sync_calls),
            "tracked_pages": int(self.stats.tracked_pages),
            "last_batch_pages": int(self.stats.last_batch_pages),
        }
