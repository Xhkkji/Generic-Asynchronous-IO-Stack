from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def _index_batch(batch, indices):
    # 从一个 batch 中抽出 miss 对应的子 batch。
    if isinstance(batch, Mapping):
        result = {}
        for key, value in batch.items():
            result[key] = _index_value(value, indices)
        return result

    if isinstance(batch, tuple):
        return tuple(_index_value(value, indices) for value in batch)

    if isinstance(batch, list):
        return [_index_value(value, indices) for value in batch]

    return _index_value(batch, indices)


def _index_value(value, indices):
    if torch.is_tensor(value):
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index_tensor)
    if isinstance(value, np.ndarray):
        return value[indices]
    if isinstance(value, list):
        return [value[idx] for idx in indices]
    if isinstance(value, tuple):
        return tuple(value[idx] for idx in indices)
    return value


def _extract_images(batch):
    if isinstance(batch, Mapping):
        return batch["images"]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


class CachedCIDSBatchLoader:
    # 第一阶段桥接层：
    # 1. 先查 host sample cache
    # 2. miss 部分再走 CIDS
    # 3. 把 miss 回来的样本放进 cache
    def __init__(self, cids_loader, sample_cache, io_mode="registered", device=None):
        if io_mode not in ("sync", "registered"):
            raise ValueError("CachedCIDSBatchLoader io_mode 只支持 sync 或 registered")
        self.cids_loader = cids_loader
        self.sample_cache = sample_cache
        self.io_mode = io_mode
        self.device = device or cids_loader.cids_device

    def fetch_batch(self, batch):
        sample_ids = self.cids_loader._sample_ids_from_batch(batch).detach().cpu().tolist()
        hit_positions = []
        hit_images = {}
        miss_positions = []
        miss_sample_ids = []

        for pos, sample_id in enumerate(sample_ids):
            cached = self.sample_cache.get(sample_id)
            if cached is None:
                miss_positions.append(pos)
                miss_sample_ids.append(sample_id)
            else:
                hit_positions.append(pos)
                hit_images[pos] = cached

        miss_images_gpu = None
        if miss_positions:
            miss_batch = _index_batch(batch, miss_positions)
            miss_return_batch = self._fetch_miss_batch(miss_batch)
            miss_images_gpu = _extract_images(miss_return_batch)
            self.sample_cache.put_many(miss_sample_ids, miss_images_gpu)

        merged_images = self._merge_images(
            batch_size=len(sample_ids),
            hit_positions=hit_positions,
            hit_images=hit_images,
            miss_positions=miss_positions,
            miss_images_gpu=miss_images_gpu,
        )
        return self.cids_loader._format_return_batch(batch, merged_images)

    def cache_summary(self):
        return self.sample_cache.summary()

    def refresh_cached_scores(self, sample_ids):
        self.sample_cache.refresh_importance(sample_ids)

    def _fetch_miss_batch(self, miss_batch):
        if self.io_mode == "sync":
            return self.cids_loader.fetch_samples_sync(miss_batch, self.device)

        request_id, _ = self.cids_loader.fetch_samples_submit_registered(miss_batch, self.device)
        self._wait_registered_request(request_id)
        return self.cids_loader.fetch_samples_get_registered(miss_batch, self.device)

    def _wait_registered_request(self, request_id):
        while True:
            ready_front = self.cids_loader.get_registered_ready_front_request_id()
            if ready_front == request_id:
                return
            returned_request_id = self.cids_loader.service_registered_poll_compatible()
            if returned_request_id == request_id:
                return

    def _merge_images(self, batch_size, hit_positions, hit_images, miss_positions, miss_images_gpu):
        result_images = [None] * batch_size

        for pos in hit_positions:
            result_images[pos] = hit_images[pos].to(self.device, non_blocking=True)

        if miss_positions:
            for local_idx, pos in enumerate(miss_positions):
                result_images[pos] = miss_images_gpu[local_idx]

        if not result_images:
            raise ValueError("merge_images 失败：空 batch")
        return torch.stack(result_images, dim=0)
