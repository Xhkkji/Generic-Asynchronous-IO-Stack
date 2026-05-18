from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import heapq

import torch


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0
    skipped_admissions: int = 0
    ghost_candidates: int = 0
    ghost_admissions: int = 0
    ghost_rejections: int = 0


@dataclass
class CachePriority:
    score: float
    freq: int
    recency: int

    def heap_key(self, sample_id):
        return (self.score, self.freq, self.recency, int(sample_id))


class LRUSampleCache:
    # 第一阶段只保留一个最简单的 host sample cache：
    # - key: sample_id
    # - value: CPU tensor
    # - eviction: 默认 LRU；若提供 importance tracker，则优先保留高 importance 样本
    def __init__(self, capacity, pin_memory=False, importance_tracker=None):
        if int(capacity) <= 0:
            raise ValueError("LRUSampleCache capacity 必须大于 0")
        self.capacity = int(capacity)
        self.pin_memory = bool(pin_memory)
        self.importance_tracker = importance_tracker
        self.ghost_score_margin = 0.5
        self._entries = OrderedDict()
        self._resident_meta = {}
        self._resident_heap = []
        self._ghost_meta = OrderedDict()
        self._tick = 0
        self.stats = CacheStats()

    def __len__(self):
        return len(self._entries)

    def contains(self, sample_id):
        return int(sample_id) in self._entries

    def get(self, sample_id):
        sample_id = int(sample_id)
        image = self._entries.get(sample_id)
        if image is None:
            self.stats.misses += 1
            return None
        self._entries.move_to_end(sample_id)
        self.stats.hits += 1
        return image

    def get_many(self, sample_ids):
        result = {}
        for sample_id in sample_ids:
            cached = self.get(sample_id)
            if cached is not None:
                result[int(sample_id)] = cached
        return result

    def put(self, sample_id, image):
        sample_id = int(sample_id)
        image = self._prepare_image(image)
        if sample_id in self._entries:
            self._entries[sample_id] = image
            self._entries.move_to_end(sample_id)
            return

        if len(self._entries) >= self.capacity:
            if not self._make_room_for(sample_id):
                self.stats.skipped_admissions += 1
                return

        self._entries[sample_id] = image
        self._admit_resident(sample_id)
        self.stats.inserts += 1

    def put_many(self, sample_ids, images):
        if torch.is_tensor(images):
            if images.dim() == 0:
                raise ValueError("put_many 的 images 不能是标量")
            image_iter = images
        else:
            image_iter = list(images)

        for sample_id, image in zip(sample_ids, image_iter):
            self.put(sample_id, image)

    def summary(self):
        return {
            "capacity": self.capacity,
            "size": len(self._entries),
            "ghost_size": len(self._ghost_meta),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "inserts": self.stats.inserts,
            "evictions": self.stats.evictions,
            "skipped_admissions": self.stats.skipped_admissions,
            "ghost_candidates": self.stats.ghost_candidates,
            "ghost_admissions": self.stats.ghost_admissions,
            "ghost_rejections": self.stats.ghost_rejections,
        }

    def refresh_importance(self, sample_ids):
        if self.importance_tracker is None:
            return

        for sample_id in sample_ids:
            sample_id = int(sample_id)
            score = self._importance_score(sample_id)
            ghost_priority = self._ghost_meta.get(sample_id)
            ghost_freq = 1 if ghost_priority is None else ghost_priority.freq + 1
            self._set_ghost_priority(
                sample_id,
                CachePriority(score=score, freq=ghost_freq, recency=self._next_recency()),
            )

            if sample_id not in self._entries:
                continue

            resident_priority = self._resident_meta.get(sample_id)
            resident_freq = 1 if resident_priority is None else resident_priority.freq + 1
            self._set_resident_priority(
                sample_id,
                CachePriority(score=score, freq=resident_freq, recency=self._next_recency()),
            )

    def _make_room_for(self, incoming_sample_id):
        if not self._entries:
            return True

        # 没有 importance tracker 时，保持原始 LRU 行为。
        if self.importance_tracker is None:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
            return True

        has_ghost_history = int(incoming_sample_id) in self._ghost_meta
        victim_sample_id, victim_priority = self._peek_eviction_candidate()
        incoming_priority = self._incoming_priority(incoming_sample_id)
        if has_ghost_history:
            self.stats.ghost_candidates += 1

        # ghost candidate 需要明显高于当前最差 resident，避免轻微优势就触发回流。
        if has_ghost_history and incoming_priority.score < victim_priority.score + self.ghost_score_margin:
            self.stats.ghost_rejections += 1
            return False

        # 新样本的重要性不高于当前最差缓存样本时，直接不准入。
        if incoming_priority.heap_key(incoming_sample_id) <= victim_priority.heap_key(victim_sample_id):
            if has_ghost_history:
                self.stats.ghost_rejections += 1
            return False

        self._entries.pop(victim_sample_id, None)
        self._evict_resident(victim_sample_id)
        self.stats.evictions += 1
        if has_ghost_history:
            self.stats.ghost_admissions += 1
        return True

    def _peek_eviction_candidate(self):
        while self._resident_heap:
            score, freq, recency, sample_id = self._resident_heap[0]
            current_priority = self._resident_meta.get(sample_id)
            if current_priority is None or sample_id not in self._entries:
                heapq.heappop(self._resident_heap)
                continue

            if current_priority.heap_key(sample_id) != (score, freq, recency, sample_id):
                heapq.heappop(self._resident_heap)
                continue

            return sample_id, current_priority

        # 理论上不该走到这里；如果发生，就退化成最老的样本。
        victim_sample_id = next(iter(self._entries))
        victim_priority = self._resident_meta.get(victim_sample_id)
        if victim_priority is None:
            victim_priority = CachePriority(
                score=self._importance_score(victim_sample_id),
                freq=0,
                recency=0,
            )
        return victim_sample_id, victim_priority

    def _importance_score(self, sample_id):
        if self.importance_tracker is None:
            return 0.0

        state = self.importance_tracker.get(sample_id)
        if state is None:
            return 0.0
        return float(state.ema_score)

    def _admit_resident(self, sample_id):
        if self.importance_tracker is None:
            return
        self._set_resident_priority(sample_id, self._incoming_priority(sample_id))

    def _evict_resident(self, sample_id):
        priority = self._resident_meta.pop(int(sample_id), None)
        if priority is None:
            return
        self._set_ghost_priority(sample_id, self._clone_priority(priority))

    def _incoming_priority(self, sample_id):
        sample_id = int(sample_id)
        ghost_priority = self._ghost_meta.get(sample_id)
        if ghost_priority is not None:
            return self._clone_priority(ghost_priority)
        return CachePriority(
            score=self._importance_score(sample_id),
            freq=0,
            recency=0,
        )

    def _set_resident_priority(self, sample_id, priority):
        sample_id = int(sample_id)
        self._resident_meta[sample_id] = priority
        heapq.heappush(self._resident_heap, priority.heap_key(sample_id))

    def _set_ghost_priority(self, sample_id, priority):
        sample_id = int(sample_id)
        self._ghost_meta[sample_id] = priority
        self._ghost_meta.move_to_end(sample_id)

    def _clone_priority(self, priority):
        return CachePriority(
            score=float(priority.score),
            freq=int(priority.freq),
            recency=self._next_recency(),
        )

    def _next_recency(self):
        self._tick += 1
        return self._tick

    def _prepare_image(self, image):
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        image = image.detach().to("cpu").contiguous()
        if self.pin_memory and image.device.type == "cpu":
            image = image.pin_memory()
        return image
