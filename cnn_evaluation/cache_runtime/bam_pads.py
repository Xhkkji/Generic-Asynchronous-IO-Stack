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
        replay_bias_scale=0.12,
        max_replace_fraction=0.002,
        locality_window=16,
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


class LogicalHotsetPADSPlanner:
    # 逻辑 hotset 调度：
    # - 主策略来源是 sample-level importance 表，而不是瞬时物理 resident 集
    # - resident lookup 只在 epoch 边界做一次低频校准，用来估计 hotset 容量
    # - 每个 batch 混入少量逻辑 hot sample，其余仍然保留随机/原始覆盖
    # - 和 shade 的靠近方式：
    #   1) hotset 仍然是主容器
    #   2) resident overlap 参与 replay 排序
    #   3) replay 优先落在短窗口内，主动制造 locality
    def __init__(
        self,
        hot_fraction=0.0625,
        max_replace_fraction=0.008,
        hotset_size_scale=1.0,
        locality_window_batches=4,
    ):
        self.hot_fraction = max(0.0, float(hot_fraction))
        self.max_replace_fraction = max(0.0, float(max_replace_fraction))
        self.hotset_size_scale = max(0.1, float(hotset_size_scale))
        self.locality_window_batches = max(0, int(locality_window_batches))
        self._hotset_ids = []

    def _target_hotset_size(self, epoch_pairs, resident_lookup):
        unique_epoch_ids = {int(sample_id) for sample_id, _ in epoch_pairs}
        resident_unique = 0
        if resident_lookup is not None:
            resident_unique = sum(
                1 for sample_id in unique_epoch_ids
                if resident_lookup.get(int(sample_id), False)
            )
        if resident_unique > 0:
            base_size = resident_unique
        elif self._hotset_ids:
            base_size = len(self._hotset_ids)
        else:
            base_size = min(len(unique_epoch_ids), 8192)
        target_size = min(
            len(unique_epoch_ids),
            max(1, int(round(base_size * self.hotset_size_scale))),
        )
        return int(target_size), int(resident_unique)

    def _rebuild_hotset(self, importance_tracker, target_size):
        if importance_tracker is None or target_size <= 0:
            self._hotset_ids = []
            return [], 0, 0
        prev_hotset = set(int(sample_id) for sample_id in self._hotset_ids)
        top_items = importance_tracker.topk(target_size)
        hotset_ids = [int(sample_id) for sample_id, _ in top_items]
        retained = sum(1 for sample_id in hotset_ids if sample_id in prev_hotset)
        newcomers = max(0, len(hotset_ids) - retained)
        self._hotset_ids = list(hotset_ids)
        return list(hotset_ids), int(retained), int(newcomers)

    def _epoch_budget(self, epoch_size, hot_count, batch_size, rep_factor):
        if epoch_size <= 0 or hot_count <= 0 or batch_size <= 0:
            return 0, 0, 0
        num_batches = (int(epoch_size) + int(batch_size) - 1) // int(batch_size)
        batch_hot_quota = max(0, int(round(int(batch_size) * self.hot_fraction)))
        quota_budget = num_batches * batch_hot_quota
        scaled_budget = int(round(quota_budget * max(0.0, float(rep_factor) - 1.0)))
        capped_budget = int(round(int(epoch_size) * self.max_replace_fraction))
        total_budget = min(int(hot_count), int(capped_budget), int(scaled_budget))
        return int(total_budget), int(batch_hot_quota), int(num_batches)

    def _score_hot_entry(self, entry):
        # hotset 内部排序：
        # - resident overlap 是加分项
        # - importance 仍然是主信号
        return (
            1 if entry["resident"] else 0,
            float(entry["score"]),
        )

    def _select_replace_positions(self, batch_pairs, importance_tracker, hotset_ids, quota):
        if quota <= 0:
            return []
        hotset_ids = set(int(sample_id) for sample_id in hotset_ids)
        ranked_positions = []
        for local_idx, (sample_id, _label) in enumerate(batch_pairs):
            sample_id = int(sample_id)
            score = _importance_score(importance_tracker, sample_id)
            ranked_positions.append(
                (
                    0 if sample_id not in hotset_ids else 1,
                    float(score),
                    int(local_idx),
                )
            )
        ranked_positions.sort(key=lambda item: (item[0], item[1], item[2]))
        return [int(local_idx) for _hot_flag, _score, local_idx in ranked_positions[:quota]]

    def _draw_bonus_pairs(self, candidate_entries, excluded_ids, quota, generator):
        if quota <= 0:
            return [], 0
        filtered = [
            entry for entry in candidate_entries
            if not entry["used"] and entry["sample_id"] not in excluded_ids
        ]
        if not filtered:
            return [], 0
        sample_count = min(int(quota), len(filtered))
        weights = torch.as_tensor(
            [max(1e-6, float(entry["score"])) for entry in filtered],
            dtype=torch.float32,
        )
        sampled_idx = torch.multinomial(
            weights,
            sample_count,
            replacement=False,
            generator=generator,
        ).view(-1).tolist()
        selected = [filtered[int(idx)] for idx in sampled_idx]
        for entry in selected:
            entry["used"] = True
        return [entry["pair"] for entry in selected], int(sample_count)

    def _draw_bonus_pairs_staged(self, hot_entries, batch_idx, excluded_ids, quota, generator):
        if quota <= 0:
            return [], {
                "resident_hot_draws": 0,
                "fallback_hot_draws": 0,
                "locality_hot_draws": 0,
            }

        local_entries = [
            entry for entry in hot_entries
            if abs(int(entry["home_batch"]) - int(batch_idx)) <= self.locality_window_batches
        ]
        local_resident = [entry for entry in local_entries if entry["resident"]]
        local_nonresident = [entry for entry in local_entries if not entry["resident"]]
        global_resident = [entry for entry in hot_entries if entry["resident"]]

        selected_pairs = []
        resident_hot_draws = 0
        fallback_hot_draws = 0
        locality_hot_draws = 0

        # staged replay：
        # 1) 优先复用窗口内且当前 resident 的 hot sample
        # 2) 再退化到窗口内其它 hot sample
        # 3) 最后才用全局 resident / 全局 hotset 兜底
        stages = [
            (local_resident, "local_resident"),
            (local_nonresident, "local_hot"),
            (global_resident, "global_resident"),
            (hot_entries, "global_hot"),
        ]
        for candidates, stage_name in stages:
            remaining = int(quota) - len(selected_pairs)
            if remaining <= 0:
                break
            bonus_pairs, draw_count = self._draw_bonus_pairs(
                candidate_entries=candidates,
                excluded_ids=excluded_ids,
                quota=remaining,
                generator=generator,
            )
            if not bonus_pairs:
                continue
            selected_pairs.extend(bonus_pairs)
            if stage_name in ("local_resident", "global_resident"):
                resident_hot_draws += int(draw_count)
            else:
                fallback_hot_draws += int(draw_count)
            if stage_name.startswith("local_"):
                locality_hot_draws += int(draw_count)

        return selected_pairs, {
            "resident_hot_draws": int(resident_hot_draws),
            "fallback_hot_draws": int(fallback_hot_draws),
            "locality_hot_draws": int(locality_hot_draws),
        }

    def build_next_epoch_plan(
        self,
        epoch_pairs,
        resident_lookup,
        importance_tracker,
        rep_factor,
        shuffle_seed,
        epoch_index,
        batch_size,
    ):
        if not epoch_pairs or importance_tracker is None:
            return None, None

        target_hotset_size, resident_unique = self._target_hotset_size(epoch_pairs, resident_lookup)
        hotset_ids, hotset_retained, hotset_newcomers = self._rebuild_hotset(
            importance_tracker=importance_tracker,
            target_size=target_hotset_size,
        )
        if not hotset_ids:
            return list(epoch_pairs), {
                "sampler": "logical_hotset",
                "resident_rate": 0.0 if resident_lookup is None else 0.0,
                "resident_count": 0,
                "nonresident_count": len(epoch_pairs),
                "resident_unique": 0,
                "nonresident_unique": len({int(sample_id) for sample_id, _ in epoch_pairs}),
                "boosted_resident": 0,
                "resident_duplicates": 0,
                "weighted_resident_draws": 0,
                "replaced_nonresident": 0,
                "batch_resident_fraction": float(self.hot_fraction),
                "batch_resident_quota": 0,
                "planned_size": len(epoch_pairs),
                "hotset_size": 0,
                "hotset_target_size": int(target_hotset_size),
                "hotset_retained": int(hotset_retained),
                "hotset_newcomers": int(hotset_newcomers),
                "hotset_resident_overlap": 0,
                "locality_window_batches": int(self.locality_window_batches),
                "locality_hot_draws": 0,
                "ghost_size": 0,
                "ghost_candidates": 0,
                "ghost_admissions": 0,
                "ghost_rejections": 0,
            }

        hotset_id_set = set(int(sample_id) for sample_id in hotset_ids)
        resident_count = sum(
            1 for sample_id, _label in epoch_pairs
            if resident_lookup is not None and resident_lookup.get(int(sample_id), False)
        )
        nonresident_count = len(epoch_pairs) - resident_count
        hot_entries = []
        for entry_idx, (sample_id, label) in enumerate(epoch_pairs):
            sample_id = int(sample_id)
            if sample_id not in hotset_id_set:
                continue
            hot_entries.append(
                {
                    "sample_id": sample_id,
                    "pair": (sample_id, int(label)),
                    "score": _importance_score(importance_tracker, sample_id),
                    "home_batch": int(entry_idx) // max(1, int(batch_size)),
                    "resident": bool(resident_lookup is not None and resident_lookup.get(sample_id, False)),
                    "used": False,
                }
            )
        hot_entries.sort(key=self._score_hot_entry, reverse=True)

        total_budget, batch_hot_quota, num_batches = self._epoch_budget(
            epoch_size=len(epoch_pairs),
            hot_count=len(hot_entries),
            batch_size=batch_size,
            rep_factor=rep_factor,
        )
        if total_budget <= 0 or batch_hot_quota <= 0:
            return list(epoch_pairs), {
                "sampler": "logical_hotset",
                "resident_rate": resident_count / max(1, len(epoch_pairs)),
                "resident_count": int(resident_count),
                "nonresident_count": int(nonresident_count),
                "resident_unique": int(resident_unique),
                "nonresident_unique": len({int(sample_id) for sample_id, _ in epoch_pairs}) - int(resident_unique),
                "boosted_resident": 0,
                "resident_duplicates": 0,
                "weighted_resident_draws": 0,
                "replaced_nonresident": 0,
                "batch_resident_fraction": float(self.hot_fraction),
                "batch_resident_quota": int(batch_hot_quota),
                "planned_size": len(epoch_pairs),
                "hotset_size": len(hotset_ids),
                "hotset_target_size": int(target_hotset_size),
                "hotset_retained": int(hotset_retained),
                "hotset_newcomers": int(hotset_newcomers),
                "hotset_resident_overlap": sum(
                    1 for sample_id in hotset_id_set
                    if resident_lookup is not None and resident_lookup.get(int(sample_id), False)
                ),
                "locality_window_batches": int(self.locality_window_batches),
                "locality_hot_draws": 0,
                "ghost_size": 0,
                "ghost_candidates": 0,
                "ghost_admissions": 0,
                "ghost_rejections": 0,
            }

        planned_pairs = list(epoch_pairs)
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
        used_budget = 0
        locality_hot_draws = 0
        resident_hot_draws = 0
        fallback_hot_draws = 0
        for batch_idx in range(num_batches):
            start = int(batch_idx) * int(batch_size)
            end = min(len(epoch_pairs), start + int(batch_size))
            batch_pairs = list(planned_pairs[start:end])
            target_used = int(round((batch_idx + 1) * total_budget / max(1, num_batches)))
            batch_budget = min(
                int(batch_hot_quota),
                max(0, target_used - used_budget),
            )
            if batch_budget <= 0:
                continue

            replace_positions = self._select_replace_positions(
                batch_pairs=batch_pairs,
                importance_tracker=importance_tracker,
                hotset_ids=hotset_id_set,
                quota=batch_budget,
            )
            if not replace_positions:
                continue

            batch_sample_ids = {int(sample_id) for sample_id, _label in batch_pairs}
            bonus_pairs, draw_summary = self._draw_bonus_pairs_staged(
                hot_entries=hot_entries,
                batch_idx=batch_idx,
                excluded_ids=batch_sample_ids,
                quota=len(replace_positions),
                generator=generator,
            )
            actual_replace = min(len(replace_positions), len(bonus_pairs))
            if actual_replace <= 0:
                continue

            locality_hot_draws += int(draw_summary["locality_hot_draws"])
            resident_hot_draws += int(draw_summary["resident_hot_draws"])
            fallback_hot_draws += int(draw_summary["fallback_hot_draws"])
            for local_pos, bonus_pair in zip(replace_positions[:actual_replace], bonus_pairs[:actual_replace]):
                batch_pairs[int(local_pos)] = bonus_pair
            planned_pairs[start:end] = batch_pairs
            used_budget += int(actual_replace)

        resident_counter = Counter(int(sample_id) for sample_id, _ in planned_pairs)
        resident_duplicates = sum(
            max(0, count - 1)
            for sample_id, count in resident_counter.items()
            if sample_id in hotset_id_set
        )
        hotset_resident_overlap = sum(
            1 for sample_id in hotset_id_set
            if resident_lookup is not None and resident_lookup.get(int(sample_id), False)
        )
        total_unique = len({int(sample_id) for sample_id, _ in epoch_pairs})
        return planned_pairs, {
            "sampler": "logical_hotset",
            "resident_rate": resident_count / max(1, len(epoch_pairs)),
            "resident_count": int(resident_count),
            "nonresident_count": int(nonresident_count),
            "resident_unique": int(resident_unique),
            "nonresident_unique": int(max(0, total_unique - resident_unique)),
            "boosted_resident": int(used_budget),
            "resident_duplicates": int(resident_duplicates),
            "weighted_resident_draws": int(used_budget),
            "replaced_nonresident": int(used_budget),
            "batch_resident_fraction": float(self.hot_fraction),
            "batch_resident_quota": int(batch_hot_quota),
            "planned_size": len(planned_pairs),
            "hotset_size": len(hotset_ids),
            "hotset_target_size": int(target_hotset_size),
            "hotset_retained": int(hotset_retained),
            "hotset_newcomers": int(hotset_newcomers),
            "hotset_resident_overlap": int(hotset_resident_overlap),
            "locality_window_batches": int(self.locality_window_batches),
            "locality_hot_draws": int(locality_hot_draws),
            "resident_hot_draws": int(resident_hot_draws),
            "fallback_hot_draws": int(fallback_hot_draws),
            "ghost_size": 0,
            "ghost_candidates": 0,
            "ghost_admissions": 0,
            "ghost_rejections": 0,
        }
