from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class GhostSampleState:
    # 轻量 ghost 元数据：
    # - 只记录最近的重要性、访问频次和最近一次出现的 epoch
    # - 不存图像，只给下一轮采样做参考
    score: float
    freq: int
    last_epoch: int


def _importance_score(importance_tracker, sample_id):
    if importance_tracker is None:
        return 0.0
    state = importance_tracker.get(sample_id)
    if state is None:
        return 0.0
    return float(state.ema_score)


class BAMPADSPlanner:
    # resident-aware PADS：
    # - 输入 resident mask + importance
    # - 输出下一轮 sample 顺序
    # - 额外维护一份 ghost metadata，近似 SHADE 的“历史 miss/evict 样本”
    def __init__(
        self,
        ghost_capacity=8192,
        ghost_score_margin=0.5,
        replacement_bias_scale=0.25,
        max_replace_fraction=0.01,
    ):
        self.ghost_capacity = max(0, int(ghost_capacity))
        self.ghost_score_margin = float(ghost_score_margin)
        self.replacement_bias_scale = max(0.0, float(replacement_bias_scale))
        self.max_replace_fraction = max(0.0, float(max_replace_fraction))
        self._ghost_meta = OrderedDict()

    def _touch_ghost(self, sample_id, score, epoch_index):
        if self.ghost_capacity <= 0:
            return
        sample_id = int(sample_id)
        prev = self._ghost_meta.get(sample_id)
        freq = 1 if prev is None else prev.freq + 1
        self._ghost_meta[sample_id] = GhostSampleState(
            score=float(score),
            freq=int(freq),
            last_epoch=int(epoch_index),
        )
        self._ghost_meta.move_to_end(sample_id)
        while len(self._ghost_meta) > self.ghost_capacity:
            self._ghost_meta.popitem(last=False)

    def _build_weighted_extras(self, resident_pairs, importance_tracker, extra_budget, shuffle_seed):
        if extra_budget <= 0 or not resident_pairs:
            return []
        weights = torch.as_tensor(
            [max(1e-6, _importance_score(importance_tracker, sample_id)) for sample_id, _ in resident_pairs],
            dtype=torch.float32,
        )
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
        sampled_idx = torch.multinomial(weights, extra_budget, replacement=True, generator=generator)
        idx_list = sampled_idx.view(-1).tolist()
        return [resident_pairs[int(i)] for i in idx_list]

    def _build_mild_bias_budget(self, epoch_size, resident_count, nonresident_count, rep_factor):
        if epoch_size <= 0 or resident_count <= 0 or nonresident_count <= 0:
            return 0
        naive_budget = max(0, int(round(resident_count * max(0.0, float(rep_factor) - 1.0))))
        scaled_budget = int(round(naive_budget * self.replacement_bias_scale))
        capped_budget = int(round(epoch_size * self.max_replace_fraction))
        budget = min(nonresident_count, scaled_budget, capped_budget)
        return max(0, budget)

    def _rank_nonresident_pairs(self, nonresident_entries, resident_pairs, importance_tracker, epoch_index):
        if not nonresident_entries:
            return [], {
                "ghost_candidates": 0,
                "ghost_admissions": 0,
                "ghost_rejections": 0,
                "ghost_size": len(self._ghost_meta),
            }

        resident_floor = 0.0
        if resident_pairs:
            resident_floor = min(
                _importance_score(importance_tracker, sample_id)
                for sample_id, _ in resident_pairs
            )

        ranked_candidates = []
        ghost_candidates = 0
        for entry_idx, sample_id, label in nonresident_entries:
            score = _importance_score(importance_tracker, sample_id)
            ghost_state = self._ghost_meta.get(int(sample_id))
            has_ghost = ghost_state is not None
            if has_ghost:
                ghost_candidates += 1
            self._touch_ghost(sample_id, score, epoch_index)

            # nonresident 排序功能：
            # - 高 importance / 有 ghost 历史的样本更应该保留
            # - 排在列表底部的样本更适合作为“轻偏置替换位”
            protect_score = score
            if has_ghost:
                protect_score += self.ghost_score_margin
            if score >= resident_floor:
                protect_score += self.ghost_score_margin
            priority = (
                protect_score,
                ghost_state.freq if ghost_state is not None else 0,
                1 if has_ghost else 0,
            )
            ranked_candidates.append(
                {
                    "index": int(entry_idx),
                    "pair": (int(sample_id), int(label)),
                    "score": float(score),
                    "has_ghost": bool(has_ghost),
                    "priority": priority,
                }
            )

        ranked_candidates.sort(key=lambda item: item["priority"], reverse=True)
        return ranked_candidates, {
            "ghost_candidates": int(ghost_candidates),
            "ghost_size": len(self._ghost_meta),
        }

    def build_next_epoch_plan(self, epoch_pairs, resident_lookup, importance_tracker, rep_factor, shuffle_seed, epoch_index):
        if not epoch_pairs or resident_lookup is None or importance_tracker is None:
            return None, None

        resident_pairs = []
        nonresident_pairs = []
        for sample_id, label in epoch_pairs:
            pair = (int(sample_id), int(label))
            if resident_lookup.get(int(sample_id), False):
                resident_pairs.append(pair)
            else:
                nonresident_pairs.append(pair)

        if not resident_pairs:
            return list(epoch_pairs), {
                "sampler": "bam_resident",
                "resident_rate": 0.0,
                "resident_count": 0,
                "nonresident_count": len(nonresident_pairs),
                "resident_unique": 0,
                "nonresident_unique": len(set(sample_id for sample_id, _ in nonresident_pairs)),
                "boosted_resident": 0,
                "weighted_resident_draws": 0,
                "ghost_size": len(self._ghost_meta),
                "ghost_candidates": 0,
                "ghost_admissions": 0,
                "ghost_rejections": 0,
                "planned_size": len(epoch_pairs),
            }

        extra_budget = self._build_mild_bias_budget(
            epoch_size=len(epoch_pairs),
            resident_count=len(resident_pairs),
            nonresident_count=len(nonresident_pairs),
            rep_factor=rep_factor,
        )
        boosted_resident = self._build_weighted_extras(
            resident_pairs=resident_pairs,
            importance_tracker=importance_tracker,
            extra_budget=extra_budget,
            shuffle_seed=shuffle_seed,
        )
        ranked_nonresident, ghost_summary = self._rank_nonresident_pairs(
            nonresident_entries=[
                (idx, sample_id, label)
                for idx, (sample_id, label) in enumerate(epoch_pairs)
                if not resident_lookup.get(int(sample_id), False)
            ],
            resident_pairs=resident_pairs,
            importance_tracker=importance_tracker,
            epoch_index=epoch_index,
        )

        planned_pairs = list(epoch_pairs)
        replace_count = min(extra_budget, len(ranked_nonresident), len(boosted_resident))
        replace_entries = ranked_nonresident[-replace_count:] if replace_count > 0 else []
        replace_entries.sort(key=lambda item: item["index"])

        ghost_admissions = 0
        ghost_rejections = 0
        for replacement, bonus_pair in zip(replace_entries, boosted_resident):
            planned_pairs[replacement["index"]] = bonus_pair
            if replacement["has_ghost"]:
                ghost_rejections += 1

        kept_nonresident_entries = ranked_nonresident[:-replace_count] if replace_count > 0 else ranked_nonresident
        for entry in kept_nonresident_entries:
            if entry["has_ghost"]:
                ghost_admissions += 1

        resident_counter = Counter(sample_id for sample_id, _ in planned_pairs)
        resident_duplicates = sum(
            max(0, count - 1)
            for sample_id, count in resident_counter.items()
            if resident_lookup.get(int(sample_id), False)
        )
        return planned_pairs, {
            "sampler": "bam_resident",
            "resident_rate": len(resident_pairs) / max(1, len(epoch_pairs)),
            "resident_count": len(resident_pairs),
            "nonresident_count": len(nonresident_pairs),
            "resident_unique": len(set(sample_id for sample_id, _ in resident_pairs)),
            "nonresident_unique": len(set(sample_id for sample_id, _ in nonresident_pairs)),
            "boosted_resident": int(replace_count),
            "resident_duplicates": int(resident_duplicates),
            "weighted_resident_draws": int(len(boosted_resident)),
            "replaced_nonresident": int(replace_count),
            "planned_size": len(planned_pairs),
            "ghost_admissions": int(ghost_admissions),
            "ghost_rejections": int(ghost_rejections),
            **ghost_summary,
        }


class ShadePADSPlanner(BAMPADSPlanner):
    # SHADE-style resident replay：
    # - 保持总样本数不变
    # - 从 resident 高重要样本中抽取 bonus 访问
    # - 在较短窗口内回放 bonus sample，主动制造 locality
    def __init__(
        self,
        ghost_capacity=8192,
        ghost_score_margin=0.5,
        replay_bias_scale=1.0,
        max_replace_fraction=0.0625,
        locality_window=32,
    ):
        super().__init__(
            ghost_capacity=ghost_capacity,
            ghost_score_margin=ghost_score_margin,
            replacement_bias_scale=replay_bias_scale,
            max_replace_fraction=max_replace_fraction,
        )
        self.replay_bias_scale = self.replacement_bias_scale
        self.locality_window = max(1, int(locality_window))

    def _build_shade_budget(self, epoch_size, resident_count, nonresident_count, rep_factor):
        if epoch_size <= 0 or resident_count <= 0 or nonresident_count <= 0:
            return 0
        naive_budget = max(0, int(round(resident_count * max(0.0, float(rep_factor) - 1.0))))
        # 收敛思路：
        # - shade 仍然比 replace 更积极
        # - 但 replay 预算先明显收窄，避免大规模改写训练分布
        scaled_budget = int(round(naive_budget * self.replay_bias_scale))
        capped_budget = int(round(epoch_size * self.max_replace_fraction))
        budget = min(nonresident_count, scaled_budget, capped_budget)
        return max(0, budget)

    def _build_diverse_extras(self, resident_pairs, importance_tracker, extra_budget, shuffle_seed):
        # replay 采样功能：
        # - 尽量覆盖更多 resident high-importance 样本
        # - 默认不放回，避免少数样本被抽到很多次
        if extra_budget <= 0 or not resident_pairs:
            return []
        sample_count = min(int(extra_budget), len(resident_pairs))
        weights = torch.as_tensor(
            [max(1e-6, _importance_score(importance_tracker, sample_id)) for sample_id, _ in resident_pairs],
            dtype=torch.float32,
        )
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
        sampled_idx = torch.multinomial(weights, sample_count, replacement=False, generator=generator)
        return [resident_pairs[int(i)] for i in sampled_idx.view(-1).tolist()]

    def _cluster_bonus_within_window(self, base_pairs, bonus_pairs):
        if not bonus_pairs:
            return list(base_pairs), 0
        bonus_counter = Counter(int(sample_id) for sample_id, _ in bonus_pairs)
        sample_to_pair = {int(sample_id): (int(sample_id), int(label)) for sample_id, label in bonus_pairs}
        planned_pairs = []
        pending_bonus = []
        locality_clusters = 0
        for idx, (sample_id, label) in enumerate(base_pairs, start=1):
            sample_id = int(sample_id)
            planned_pairs.append((sample_id, int(label)))
            bonus_count = bonus_counter.pop(sample_id, 0)
            if bonus_count > 0:
                locality_clusters += 1
                pending_bonus.extend([sample_to_pair[sample_id]] * bonus_count)
            # locality 形状：
            # - 不再紧贴第一次访问后立刻回放
            # - 改成短窗口内冲刷一次 pending bonus，减轻局部簇状访问
            if pending_bonus and idx % self.locality_window == 0:
                planned_pairs.extend(pending_bonus)
                pending_bonus = []
        if pending_bonus:
            planned_pairs.extend(pending_bonus)
        return planned_pairs, int(locality_clusters)

    def build_next_epoch_plan(
        self,
        epoch_pairs,
        resident_lookup,
        importance_tracker,
        rep_factor,
        shuffle_seed,
        epoch_index,
    ):
        if not epoch_pairs or resident_lookup is None or importance_tracker is None:
            return None, None

        resident_pairs = []
        nonresident_pairs = []
        for sample_id, label in epoch_pairs:
            pair = (int(sample_id), int(label))
            if resident_lookup.get(int(sample_id), False):
                resident_pairs.append(pair)
            else:
                nonresident_pairs.append(pair)

        if not resident_pairs or not nonresident_pairs:
            return list(epoch_pairs), {
                "sampler": "shade",
                "resident_rate": len(resident_pairs) / max(1, len(epoch_pairs)),
                "resident_count": len(resident_pairs),
                "nonresident_count": len(nonresident_pairs),
                "resident_unique": len(set(sample_id for sample_id, _ in resident_pairs)),
                "nonresident_unique": len(set(sample_id for sample_id, _ in nonresident_pairs)),
                "boosted_resident": 0,
                "resident_duplicates": 0,
                "weighted_resident_draws": 0,
                "replaced_nonresident": 0,
                "ghost_size": len(self._ghost_meta),
                "ghost_candidates": 0,
                "ghost_admissions": 0,
                "ghost_rejections": 0,
                "planned_size": len(epoch_pairs),
            }

        replay_budget = self._build_shade_budget(
            epoch_size=len(epoch_pairs),
            resident_count=len(resident_pairs),
            nonresident_count=len(nonresident_pairs),
            rep_factor=rep_factor,
        )
        if replay_budget <= 0:
            return list(epoch_pairs), {
                "sampler": "shade",
                "resident_rate": len(resident_pairs) / max(1, len(epoch_pairs)),
                "resident_count": len(resident_pairs),
                "nonresident_count": len(nonresident_pairs),
                "resident_unique": len(set(sample_id for sample_id, _ in resident_pairs)),
                "nonresident_unique": len(set(sample_id for sample_id, _ in nonresident_pairs)),
                "boosted_resident": 0,
                "resident_duplicates": 0,
                "weighted_resident_draws": 0,
                "replaced_nonresident": 0,
                "ghost_size": len(self._ghost_meta),
                "ghost_candidates": 0,
                "ghost_admissions": 0,
                "ghost_rejections": 0,
                "planned_size": len(epoch_pairs),
            }

        boosted_resident = self._build_diverse_extras(
            resident_pairs=resident_pairs,
            importance_tracker=importance_tracker,
            extra_budget=replay_budget,
            shuffle_seed=shuffle_seed,
        )
        ranked_nonresident, ghost_summary = self._rank_nonresident_pairs(
            nonresident_entries=[
                (idx, sample_id, label)
                for idx, (sample_id, label) in enumerate(epoch_pairs)
                if not resident_lookup.get(int(sample_id), False)
            ],
            resident_pairs=resident_pairs,
            importance_tracker=importance_tracker,
            epoch_index=epoch_index,
        )

        replace_count = min(replay_budget, len(ranked_nonresident), len(boosted_resident))
        replace_entries = ranked_nonresident[-replace_count:] if replace_count > 0 else []
        replace_indices = {item["index"] for item in replace_entries}
        ghost_rejections = sum(1 for item in replace_entries if item["has_ghost"])
        ghost_admissions = sum(
            1 for item in ranked_nonresident
            if item["index"] not in replace_indices and item["has_ghost"]
        )

        base_pairs = [
            pair
            for idx, pair in enumerate(epoch_pairs)
            if idx not in replace_indices
        ]
        clustered_bonus = boosted_resident[:replace_count]
        planned_pairs, locality_clusters = self._cluster_bonus_within_window(
            base_pairs=base_pairs,
            bonus_pairs=clustered_bonus,
        )

        resident_counter = Counter(sample_id for sample_id, _ in planned_pairs)
        resident_duplicates = sum(
            max(0, count - 1)
            for sample_id, count in resident_counter.items()
            if resident_lookup.get(int(sample_id), False)
        )
        return planned_pairs, {
            "sampler": "shade",
            "resident_rate": len(resident_pairs) / max(1, len(epoch_pairs)),
            "resident_count": len(resident_pairs),
            "nonresident_count": len(nonresident_pairs),
            "resident_unique": len(set(sample_id for sample_id, _ in resident_pairs)),
            "nonresident_unique": len(set(sample_id for sample_id, _ in nonresident_pairs)),
            "boosted_resident": int(replace_count),
            "resident_duplicates": int(resident_duplicates),
            "weighted_resident_draws": int(len(boosted_resident)),
            "replaced_nonresident": int(replace_count),
            "ghost_size": len(self._ghost_meta),
            "ghost_candidates": int(ghost_summary.get("ghost_candidates", 0)),
            "ghost_admissions": int(ghost_admissions),
            "ghost_rejections": int(ghost_rejections),
            "planned_size": len(planned_pairs),
            "locality_clusters": int(locality_clusters),
            "locality_window": int(self.locality_window),
        }
