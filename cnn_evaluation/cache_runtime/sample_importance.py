from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SampleImportanceState:
    # 第一阶段只记录最核心的几项：
    # - seen_count: 被看过多少次
    # - last_loss: 最近一次 loss
    # - ema_loss: loss 的指数滑动平均
    # - last_step: 最近一次被访问的 step
    seen_count: int = 0
    last_loss: float = 0.0
    ema_loss: float = 0.0
    last_step: int = -1


class SampleImportanceTracker:
    # 最小版 per-sample importance 跟踪器。
    # 当前 importance 直接定义为 ema_loss，先不引入更复杂的 rank/policy。
    def __init__(self, ema_alpha=0.9):
        ema_alpha = float(ema_alpha)
        if not 0.0 <= ema_alpha < 1.0:
            raise ValueError("ema_alpha 必须满足 0 <= ema_alpha < 1")
        self.ema_alpha = ema_alpha
        self._states = {}

    def __len__(self):
        return len(self._states)

    def update(self, sample_ids, sample_losses, step):
        if not torch.is_tensor(sample_ids):
            sample_ids = torch.as_tensor(sample_ids)
        if not torch.is_tensor(sample_losses):
            sample_losses = torch.as_tensor(sample_losses)

        sample_ids = sample_ids.detach().cpu().view(-1).tolist()
        sample_losses = sample_losses.detach().cpu().view(-1).tolist()

        for sample_id, sample_loss in zip(sample_ids, sample_losses):
            sample_id = int(sample_id)
            sample_loss = float(sample_loss)
            state = self._states.get(sample_id)
            if state is None:
                state = SampleImportanceState(
                    seen_count=1,
                    last_loss=sample_loss,
                    ema_loss=sample_loss,
                    last_step=int(step),
                )
                self._states[sample_id] = state
                continue

            state.seen_count += 1
            state.last_loss = sample_loss
            state.ema_loss = (
                self.ema_alpha * state.ema_loss
                + (1.0 - self.ema_alpha) * sample_loss
            )
            state.last_step = int(step)

    def get(self, sample_id):
        return self._states.get(int(sample_id))

    def topk(self, k=5):
        k = max(0, int(k))
        items = sorted(
            self._states.items(),
            key=lambda item: item[1].ema_loss,
            reverse=True,
        )
        return items[:k]

    def summary(self, topk=5):
        tracked = len(self._states)
        if tracked == 0:
            return {
                "tracked_samples": 0,
                "avg_ema_loss": 0.0,
                "max_ema_loss": 0.0,
                "topk": [],
            }

        ema_losses = [state.ema_loss for state in self._states.values()]
        top_items = self.topk(topk)
        return {
            "tracked_samples": tracked,
            "avg_ema_loss": sum(ema_losses) / tracked,
            "max_ema_loss": max(ema_losses),
            "topk": [
                {
                    "sample_id": int(sample_id),
                    "ema_loss": float(state.ema_loss),
                    "seen_count": int(state.seen_count),
                }
                for sample_id, state in top_items
            ],
        }
