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
        self._entries = OrderedDict()
        self._importance_heap = []
        self._recency = {}
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
        self._mark_recent(sample_id)
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
            self._mark_recent(sample_id)
            return

        if len(self._entries) >= self.capacity:
            if not self._make_room_for(sample_id):
                self.stats.skipped_admissions += 1
                return

        self._entries[sample_id] = image
        self._mark_recent(sample_id)
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
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "inserts": self.stats.inserts,
            "evictions": self.stats.evictions,
            "skipped_admissions": self.stats.skipped_admissions,
        }

    def refresh_importance(self, sample_ids):
        if self.importance_tracker is None:
            return

        for sample_id in sample_ids:
            sample_id = int(sample_id)
            if sample_id not in self._entries:
                continue
            recency = self._recency.get(sample_id)
            if recency is None:
                continue
            heapq.heappush(
                self._importance_heap,
                (self._importance_score(sample_id), recency, sample_id),
            )

    def _make_room_for(self, incoming_sample_id):
        if not self._entries:
            return True

        # 没有 importance tracker 时，保持原始 LRU 行为。
        if self.importance_tracker is None:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
            return True

        victim_sample_id, victim_score = self._peek_eviction_candidate()
        incoming_score = self._importance_score(incoming_sample_id)

        # 新样本的重要性不高于当前最差缓存样本时，直接不准入。
        if incoming_score <= victim_score:
            return False

        self._entries.pop(victim_sample_id, None)
        self._recency.pop(victim_sample_id, None)
        self.stats.evictions += 1
        return True

    def _peek_eviction_candidate(self):
        while self._importance_heap:
            score, recency, sample_id = self._importance_heap[0]
            if sample_id not in self._entries:
                heapq.heappop(self._importance_heap)
                continue

            current_recency = self._recency.get(sample_id)
            if current_recency != recency:
                heapq.heappop(self._importance_heap)
                continue

            current_score = self._importance_score(sample_id)
            if current_score != score:
                heapq.heapreplace(
                    self._importance_heap,
                    (current_score, current_recency, sample_id),
                )
                continue

            return sample_id, score

        # 理论上不该走到这里；如果发生，就退化成最老的样本。
        victim_sample_id = next(iter(self._entries))
        return victim_sample_id, self._importance_score(victim_sample_id)

    def _importance_score(self, sample_id):
        if self.importance_tracker is None:
            return 0.0

        state = self.importance_tracker.get(sample_id)
        if state is None:
            return 0.0
        return float(state.ema_loss)

    def _mark_recent(self, sample_id):
        if self.importance_tracker is None:
            return

        self._tick += 1
        recency = self._tick
        self._recency[sample_id] = recency
        heapq.heappush(
            self._importance_heap,
            (self._importance_score(sample_id), recency, sample_id),
        )

    def _prepare_image(self, image):
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        image = image.detach().to("cpu").contiguous()
        if self.pin_memory and image.device.type == "cpu":
            image = image.pin_memory()
        return image
