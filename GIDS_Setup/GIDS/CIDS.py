import math
import os
import time
import json
from collections.abc import Mapping

import torch
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader, Dataset
import numpy as np

import BAM_Feature_Store


def _get_device(device):
    # 统一规范 device，和 GIDS 保持一致。
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _dtype_name_to_torch_dtype(dtype_name):
    # 把 meta.json 里的 dtype 名字映射成 torch dtype。
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "uint8": torch.uint8,
        "int64": torch.long,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持的 CIDS dtype: {dtype_name}")
    return mapping[dtype_name]


def _dtype_name_to_numpy_dtype(dtype_name):
    # 把 meta.json 里的 dtype 名字映射成 numpy dtype。
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
        "uint8": np.uint8,
        "int64": np.int64,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持的 CIDS numpy dtype: {dtype_name}")
    return mapping[dtype_name]


def _prepared_array_to_training_float32(array, dtype_name):
    # 把 prepared dataset 中的一条样本数据统一转换成训练语义的 float32：
    # - uint8: 视为 0~255 像素，缩放到 0~1
    # - float16/float32: 保持原值，仅转成 float32 便于后续统一处理
    if dtype_name == "uint8":
        return array.astype(np.float32, copy=False) / 255.0
    if array.dtype != np.float32:
        return array.astype(np.float32, copy=False)
    return array


def _images_to_training_tensor(images, dtype_name):
    # 把底层读回的存储张量统一转成训练语义。
    if dtype_name == "uint8":
        return images.to(torch.float32).div_(255.0)
    return images


class _PrefetchingIter_async_registered_try_service_cids(object):
    # CIDS 的最小 registered try-service 迭代器：
    # 按 sample_id 提交图片读取，请求完成后再把图片 batch 返回给训练。
    REQUEST_STATE_SUBMITTED = 0
    REQUEST_STATE_READY = 1
    REQUEST_STATE_CONSUMED = 2

    def __init__(self, dataloader, dataloader_it, CIDS_Loader=None, prefetch_depth=1,
                 registered_split=1):
        self.dataloader = dataloader
        self.dataloader_it = dataloader_it
        self.CIDS_Loader = CIDS_Loader

        self.prefetch_queue = []
        self.exhausted = False
        self.prefetch_depth = max(1, int(prefetch_depth))
        self.registered_split = max(1, int(registered_split))
        self.enable_skip_front = bool(getattr(
            self.CIDS_Loader, "registered_enable_skip_front", True))
        self.try_window_size = max(1, int(getattr(
            self.CIDS_Loader, "registered_try_window_size", 1)))
        self.debug = os.environ.get(
            "CIDS_DEBUG", "0") not in ("0", "", "false", "False")
        self.profile_gpu_timing = os.environ.get(
            "CIDS_PROFILE_GPU_TIMING", "0") not in ("0", "", "false", "False")
        self.trace_registered_calls = os.environ.get(
            "CIDS_REGISTERED_TRACE_CALLS", "0") not in ("0", "", "false", "False")
        # registered front-request 超时保护：
        # 如果底层 request 长时间不进入 READY，就主动报错并打印状态，
        # 避免训练线程无限挂在 next(data_iter) 上。
        self.poll_timeout_sec = float(os.environ.get(
            "CIDS_REGISTERED_POLL_TIMEOUT_SEC", "60"))
        self.poll_log_interval = max(1, int(os.environ.get(
            "CIDS_REGISTERED_POLL_LOG_INTERVAL", "128")))

        self._warmup_prefetch_queue()

    def _debug_log(self, message):
        # 统一调试输出入口，默认关闭。
        if self.debug:
            print(f"[CIDS] {message}", flush=True)

    def _trace_registered_log(self, message):
        # registered 关键调用跟踪：
        # 打开后能区分是卡在 poll 还是卡在 get。
        if self.trace_registered_calls:
            print(f"[CIDS_REGISTERED_TRACE] {message}", flush=True)

    def _sync_for_profile_timing(self):
        # 仅在 profiling 时把异步 GPU 工作结算到当前阶段区间里，便于看到 submit/poll/get 的 GPU 时间。
        device = _get_device(self.CIDS_Loader.cids_device)
        if self.profile_gpu_timing and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _warmup_prefetch_queue(self):
        # 初始化时先把预取队列填到目标深度。
        while len(self.prefetch_queue) < self.prefetch_depth and not self.exhausted:
            if not self._submit_prefetch():
                break

    def _slice_batch(self, batch, start, end):
        # 沿 batch 维切出一个 sub-batch，供 split submit 使用。
        if isinstance(batch, Mapping):
            result = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    result[key] = value[start:end]
                elif isinstance(value, np.ndarray):
                    result[key] = value[start:end]
                elif isinstance(value, (list, tuple)):
                    result[key] = value[start:end]
                else:
                    result[key] = value
            return result

        if isinstance(batch, tuple):
            items = []
            for value in batch:
                if torch.is_tensor(value):
                    items.append(value[start:end])
                elif isinstance(value, np.ndarray):
                    items.append(value[start:end])
                elif isinstance(value, (list, tuple)):
                    items.append(value[start:end])
                else:
                    items.append(value)
            return tuple(items)

        if isinstance(batch, list):
            items = []
            for value in batch:
                if torch.is_tensor(value):
                    items.append(value[start:end])
                elif isinstance(value, np.ndarray):
                    items.append(value[start:end])
                elif isinstance(value, (list, tuple)):
                    items.append(value[start:end])
                else:
                    items.append(value)
            return items

        if torch.is_tensor(batch):
            return batch[start:end]
        if isinstance(batch, np.ndarray):
            return batch[start:end]
        if isinstance(batch, (list, tuple)):
            return batch[start:end]
        return batch

    def _split_batch(self, batch):
        # 把一个训练 batch 按 sample 维切成多个 sub-batch request。
        sample_ids = self.CIDS_Loader._sample_ids_from_batch(batch)
        batch_size = int(sample_ids.numel())
        if batch_size == 0:
            return [batch]

        split = min(self.registered_split, batch_size)
        if split <= 1:
            return [batch]

        chunk_size = math.ceil(batch_size / split)
        sub_batches = []
        for start in range(0, batch_size, chunk_size):
            end = min(batch_size, start + chunk_size)
            sub_batches.append(self._slice_batch(batch, start, end))
        return sub_batches

    def _merge_return_batches(self, merged_batches):
        # 把多个 sub-batch 的返回结果重新拼回一个完整训练 batch。
        if not merged_batches:
            raise ValueError("merge_return_batches 需要至少一个 sub-batch")
        if len(merged_batches) == 1:
            return merged_batches[0]

        first = merged_batches[0]
        if isinstance(first, Mapping):
            result = {}
            for key in first.keys():
                values = [item[key] for item in merged_batches]
                if torch.is_tensor(values[0]):
                    result[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], np.ndarray):
                    result[key] = np.concatenate(values, axis=0)
                elif isinstance(values[0], list):
                    merged = []
                    for value in values:
                        merged.extend(value)
                    result[key] = merged
                elif isinstance(values[0], tuple):
                    merged = []
                    for value in values:
                        merged.extend(list(value))
                    result[key] = tuple(merged)
                else:
                    result[key] = values
            return result

        if isinstance(first, tuple):
            merged_fields = []
            for field_idx in range(len(first)):
                values = [item[field_idx] for item in merged_batches]
                if torch.is_tensor(values[0]):
                    merged_fields.append(torch.cat(values, dim=0))
                elif isinstance(values[0], np.ndarray):
                    merged_fields.append(np.concatenate(values, axis=0))
                elif isinstance(values[0], list):
                    merged = []
                    for value in values:
                        merged.extend(value)
                    merged_fields.append(merged)
                elif isinstance(values[0], tuple):
                    merged = []
                    for value in values:
                        merged.extend(list(value))
                    merged_fields.append(tuple(merged))
                else:
                    merged_fields.append(values)
            return tuple(merged_fields)

        if isinstance(first, list):
            merged_fields = []
            for field_idx in range(len(first)):
                values = [item[field_idx] for item in merged_batches]
                if torch.is_tensor(values[0]):
                    merged_fields.append(torch.cat(values, dim=0))
                elif isinstance(values[0], np.ndarray):
                    merged_fields.append(np.concatenate(values, axis=0))
                elif isinstance(values[0], list):
                    merged = []
                    for value in values:
                        merged.extend(value)
                    merged_fields.append(merged)
                elif isinstance(values[0], tuple):
                    merged = []
                    for value in values:
                        merged.extend(list(value))
                    merged_fields.append(merged)
                else:
                    merged_fields.append(values)
            return merged_fields

        if torch.is_tensor(first):
            return torch.cat(merged_batches, dim=0)
        if isinstance(first, np.ndarray):
            return np.concatenate(merged_batches, axis=0)
        if isinstance(first, list):
            merged = []
            for item in merged_batches:
                merged.extend(item)
            return merged
        return merged_batches

    def _submit_prefetch(self):
        # 从底层 dataloader 取一个 batch，并提交对应图片读取请求。
        if self.exhausted:
            return False

        try:
            next_batch = next(self.dataloader_it)
        except StopIteration:
            self.exhausted = True
            return False

        split_batches = self._split_batch(next_batch)
        request_ids = []
        row_indices_refs = []
        for split_batch in split_batches:
            with record_function("cids.registered.submit"):
                request_id, row_indices = self.CIDS_Loader.fetch_samples_submit_registered(
                    split_batch, self.CIDS_Loader.cids_device)
                self._sync_for_profile_timing()
            request_ids.append(request_id)
            row_indices_refs.append(row_indices)

        self.prefetch_queue.append({
            "batch": next_batch,
            "split_batches": split_batches,
            "request_ids": request_ids,
            "row_indices_refs": row_indices_refs,
        })
        self._debug_log(
            f"提交完成 request_ids={request_ids}, split={len(request_ids)}, queue_len={len(self.prefetch_queue)}")
        return True

    def _service_prefetch_nonblocking(self):
        # 对 front 后面的 request 做一次轻量预热。
        if not self.enable_skip_front:
            return
        if len(self.prefetch_queue) <= 1:
            return
        if self.CIDS_Loader.get_registered_outstanding_count() <= 1:
            return
        self.CIDS_Loader.service_registered_try_poll_window_skip_front(
            self.try_window_size)

    def _registered_queue_snapshot(self, max_entries=8):
        # 抓一份 outstanding 队列摘要，方便超时后定位卡住的 request。
        outstanding = int(self.CIDS_Loader.get_registered_outstanding_count())
        ready_front = int(self.CIDS_Loader.get_registered_ready_front_request_id())
        entries = []
        for offset in range(min(max_entries, max(0, outstanding))):
            try:
                request_id = int(self.CIDS_Loader.get_registered_request_id_at(offset))
                state = int(self.CIDS_Loader.get_registered_request_state_at(offset))
            except Exception as exc:
                entries.append({
                    "offset": offset,
                    "error": repr(exc),
                })
                break
            entries.append({
                "offset": offset,
                "request_id": request_id,
                "state": state,
            })
        return {
            "outstanding": outstanding,
            "ready_front": ready_front,
            "entries": entries,
        }

    def _poll_request_ready(self, expected_request_id):
        # 只等待当前 front 的一个 sub-request ready。
        if expected_request_id == 0:
            return
        ready_front = self.CIDS_Loader.get_registered_ready_front_request_id()
        if ready_front == expected_request_id:
            return
        self._debug_log(
            f"开始等待 sub-request ready expected={expected_request_id} "
            f"outstanding={self.CIDS_Loader.get_registered_outstanding_count()} "
            f"ready_front={ready_front}"
        )
        try_loops = 0
        wait_start_time = time.perf_counter()
        while True:
            # 对当前 front sub-request，始终优先走 compatible poll。
            # rowctx registered 的 front request 需要这个路径主动把 page 从
            # SUBMITTED/NV_NB 推进到 READY；try_poll 只做 completion/ready 检查，
            # 在 outstanding>1 时单独用它容易一直卡住。
            with record_function("cids.registered.poll"):
                self._trace_registered_log(
                    f"phase=poll_enter expected_request_id={expected_request_id} "
                    f"ready_front={self.CIDS_Loader.get_registered_ready_front_request_id()} "
                    f"outstanding={self.CIDS_Loader.get_registered_outstanding_count()} "
                    f"loops={try_loops}"
                )
                request_id = self.CIDS_Loader.service_registered_poll_compatible()
                self._trace_registered_log(
                    f"phase=poll_return expected_request_id={expected_request_id} "
                    f"returned_request_id={request_id} "
                    f"ready_front={self.CIDS_Loader.get_registered_ready_front_request_id()} "
                    f"outstanding={self.CIDS_Loader.get_registered_outstanding_count()} "
                    f"loops={try_loops + 1}"
                )
                self._sync_for_profile_timing()
            try_loops += 1
            if try_loops == 1 or try_loops % self.poll_log_interval == 0:
                self._debug_log(
                    f"等待中 expected={expected_request_id} "
                    f"returned={request_id} "
                    f"ready_front={self.CIDS_Loader.get_registered_ready_front_request_id()} "
                    f"outstanding={self.CIDS_Loader.get_registered_outstanding_count()} "
                    f"loops={try_loops}"
                )
            if try_loops % 8 == 0:
                self._service_prefetch_nonblocking()
            ready_front = self.CIDS_Loader.get_registered_ready_front_request_id()
            if request_id == expected_request_id or ready_front == expected_request_id:
                self._debug_log(
                    f"等待完成 expected={expected_request_id} "
                    f"returned={request_id} ready_front={ready_front} loops={try_loops}"
                )
                return
            elapsed = time.perf_counter() - wait_start_time
            if elapsed >= self.poll_timeout_sec:
                snapshot = self._registered_queue_snapshot()
                raise RuntimeError(
                    "registered front request timed out while waiting for READY: "
                    f"expected_request_id={expected_request_id} "
                    f"last_returned_request_id={request_id} "
                    f"elapsed_sec={elapsed:.2f} "
                    f"queue_snapshot={snapshot}"
                )

    def __iter__(self):
        return self

    def __next__(self):
        # 返回一个完整的 CNN batch：(images, labels, ...)
        if not self.prefetch_queue and self.exhausted:
            raise StopIteration

        self._service_prefetch_nonblocking()
        next_item = self.prefetch_queue.pop(0)
        split_batches = []
        for split_batch, request_id in zip(next_item["split_batches"], next_item["request_ids"]):
            self._debug_log(f"准备消费 sub-request request_id={request_id}")
            self._poll_request_ready(request_id)
            with record_function("cids.registered.get"):
                self._trace_registered_log(
                    f"phase=get_enter request_id={request_id} "
                    f"ready_front={self.CIDS_Loader.get_registered_ready_front_request_id()} "
                    f"outstanding={self.CIDS_Loader.get_registered_outstanding_count()}"
                )
                split_batches.append(
                    self.CIDS_Loader.fetch_samples_get_registered(
                        split_batch, self.CIDS_Loader.cids_device)
                )
                self._trace_registered_log(
                    f"phase=get_return request_id={request_id} "
                    f"ready_front={self.CIDS_Loader.get_registered_ready_front_request_id()} "
                    f"outstanding={self.CIDS_Loader.get_registered_outstanding_count()}"
                )
                self._sync_for_profile_timing()
            self._debug_log(f"完成消费 sub-request request_id={request_id}")
            self._service_prefetch_nonblocking()
            while len(self.prefetch_queue) < self.prefetch_depth and not self.exhausted:
                if not self._submit_prefetch():
                    break
        with record_function("cids.registered.merge"):
            batch = self._merge_return_batches(split_batches)

        while len(self.prefetch_queue) < self.prefetch_depth and not self.exhausted:
            if not self._submit_prefetch():
                break

        return batch


class _PrefetchingIter_sync_cids(object):
    # CIDS 的最原始同步读取路径：
    # 每次直接读取一个 batch，不做 submit/poll/get 拆分。
    def __init__(self, dataloader_it, CIDS_Loader=None):
        self.dataloader_it = dataloader_it
        self.CIDS_Loader = CIDS_Loader
        self.profile_gpu_timing = os.environ.get(
            "CIDS_PROFILE_GPU_TIMING", "0") not in ("0", "", "false", "False")

    def __iter__(self):
        return self

    def __next__(self):
        next_batch = next(self.dataloader_it)
        with record_function("cids.sync.read"):
            batch = self.CIDS_Loader.fetch_samples_sync(
                next_batch, self.CIDS_Loader.cids_device)
            device = _get_device(self.CIDS_Loader.cids_device)
            if self.profile_gpu_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            return batch


class CIDS_DataLoader(DataLoader):
    # 面向 CNN/image 训练的 DataLoader 包装器：
    # 底层 dataset 只需要提供 sample_id 和 label。
    def __init__(self, *args, CIDS_Loader=None, prefetch_depth=None, io_mode="registered",
                 registered_split=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.CIDS_Loader = CIDS_Loader
        self.prefetch_depth = int(prefetch_depth or getattr(
            self.CIDS_Loader, "iter_prefetch_depth", 1))
        self.io_mode = io_mode
        self.registered_split = int(registered_split or getattr(
            self.CIDS_Loader, "registered_split", 1))

    def __iter__(self):
        # 目前只保留两个最小分支：
        # sync：最原始同步直读
        # registered：异步 registered try-service
        if self.io_mode == "sync":
            return _PrefetchingIter_sync_cids(
                super().__iter__(),
                self.CIDS_Loader,
            )
        if self.io_mode != "registered":
            raise ValueError(f"不支持的 CIDS io_mode: {self.io_mode}")
        return _PrefetchingIter_async_registered_try_service_cids(
            self,
            super().__iter__(),
            self.CIDS_Loader,
            prefetch_depth=self.prefetch_depth,
            registered_split=self.registered_split,
        )


class CIDSSampleDataset(Dataset):
    # 最小 sample_id 数据集：
    # 每个样本只返回 sample_id 和 label，真正图片内容由 CIDS 从 SSD 读取。
    def __init__(self, sample_ids, labels):
        if len(sample_ids) != len(labels):
            raise ValueError("sample_ids 和 labels 长度必须一致")
        self.sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return self.sample_ids[index], self.labels[index]


class CIDSPreparedDataset(CIDSSampleDataset):
    # 直接读取 cids_prepare_dataset.py 生成的目录：
    # 自动加载 labels.npy 和 meta.json，返回 sample_id + label。
    def __init__(self, prepared_root, start_sample_id=0):
        meta_path = os.path.join(prepared_root, "meta.json")
        labels_path = os.path.join(prepared_root, "labels.npy")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"未找到 CIDS meta.json: {meta_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"未找到 CIDS labels.npy: {labels_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        labels = np.load(labels_path)
        sample_ids = list(range(start_sample_id, start_sample_id + len(labels)))
        super().__init__(sample_ids, labels)

        self.prepared_root = prepared_root
        self.meta = meta
        self.num_classes = len(meta.get("classes", []))
        self.classes = meta.get("classes", [])
        self.class_to_idx = meta.get("class_to_idx", {})


class CIDSImageFolderDataset(CIDSSampleDataset):
    # 通用 ImageFolder 索引数据集：
    # 只负责把目录结构映射成 sample_id/label，不直接读取图片内容。
    def __init__(self, root, start_sample_id=0):
        try:
            from torchvision.datasets import ImageFolder
        except ImportError as exc:
            raise ImportError("使用 CIDSImageFolderDataset 需要 torchvision") from exc

        image_folder = ImageFolder(root)
        labels = [label for _, label in image_folder.samples]
        sample_ids = list(range(start_sample_id, start_sample_id + len(labels)))
        super().__init__(sample_ids, labels)
        self.root = root
        self.classes = image_folder.classes
        self.class_to_idx = image_folder.class_to_idx
        self.num_classes = len(self.classes)


class CIDSTinyImageNetDataset(CIDSSampleDataset):
    # Tiny ImageNet 的最小索引数据集：
    # 训练集走标准 train 子目录；验证集按 val_annotations.txt 建立标签。
    def __init__(self, root, split="train", start_sample_id=0):
        split = split.lower()
        if split not in ("train", "val"):
            raise ValueError("Tiny ImageNet split 只支持 train 或 val")

        wnids_path = os.path.join(root, "wnids.txt")
        if not os.path.exists(wnids_path):
            raise FileNotFoundError(f"未找到 Tiny ImageNet 的 wnids.txt: {wnids_path}")

        with open(wnids_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        labels = []
        if split == "train":
            train_root = os.path.join(root, "train")
            for cls_name in classes:
                img_dir = os.path.join(train_root, cls_name, "images")
                if not os.path.isdir(img_dir):
                    continue
                image_names = sorted(
                    name for name in os.listdir(img_dir)
                    if not name.startswith("."))
                labels.extend([class_to_idx[cls_name]] * len(image_names))
        else:
            val_root = os.path.join(root, "val")
            anno_path = os.path.join(val_root, "val_annotations.txt")
            if not os.path.exists(anno_path):
                raise FileNotFoundError(
                    f"未找到 Tiny ImageNet 验证集标注文件: {anno_path}")
            with open(anno_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    cls_name = parts[1]
                    if cls_name not in class_to_idx:
                        continue
                    labels.append(class_to_idx[cls_name])

        sample_ids = list(range(start_sample_id, start_sample_id + len(labels)))
        super().__init__(sample_ids, labels)
        self.root = root
        self.split = split
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.num_classes = len(classes)


class CIDSImageNet1kDataset(CIDSImageFolderDataset):
    # ImageNet-1K 的最小索引数据集：
    # 假设已经整理成 ImageFolder 兼容目录，如 train/<class>/*.JPEG。
    def __init__(self, root, split="train", start_sample_id=0):
        split = split.lower()
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(
                f"未找到 ImageNet-1K {split} 目录: {split_root}")
        super().__init__(split_root, start_sample_id=start_sample_id)
        self.dataset_root = root
        self.split = split


class CIDS(object):
    # CIDS 的最小骨架：
    # 先假设图片已经离线整理成固定尺寸、固定长度的 sample tensor，
    # 每个 sample_id 对应 storage 上的一整行。
    def __init__(
        self,
        sample_shape=None,
        page_size=4096,
        off=0,
        num_ele=300 * 1000 * 1000 * 1024,
        num_ssd=1,
        ssd_list=None,
        cache_size=10,
        ctrl_idx=0,
        long_type=False,
        prepared_root=None,
        meta_path=None,
        dtype_name=None,
    ):
        # 优先从整理后的 meta.json 自动读取 sample 配置，避免手工写死 shape。
        self.prepared_root = prepared_root
        if meta_path is None and prepared_root is not None:
            meta_path = os.path.join(prepared_root, "meta.json")
        self.meta_path = meta_path
        self.meta = None

        if self.meta_path is not None:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            sample_shape = tuple(self.meta["shape"])
            dtype_name = self.meta["dtype"]

        if sample_shape is None:
            raise ValueError("CIDS 需要 sample_shape，或通过 prepared_root/meta_path 自动读取")
        if dtype_name is None:
            dtype_name = "float16"

        if long_type:
            self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store_long()
            self.return_dtype = torch.long
            self.storage_itemsize = 8
        elif dtype_name == "uint8":
            self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store_byte()
            self.return_dtype = torch.uint8
            self.storage_itemsize = 1
        else:
            self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store_float()
            self.return_dtype = torch.float32
            self.storage_itemsize = 4

        self.images_path = None
        self.labels_path = None
        if self.meta is not None and self.prepared_root is not None:
            self.images_path = os.path.join(
                self.prepared_root, self.meta.get("images_file", "images.bin"))
            self.labels_path = os.path.join(
                self.prepared_root, self.meta.get("labels_file", "labels.npy"))

        self.page_size = page_size
        self.sample_shape = tuple(sample_shape)
        self.sample_dtype_name = dtype_name
        self.sample_dim = int(math.prod(self.sample_shape))
        # 1 张图片不会直接当成 1 行，而是拆成若干个连续 row。
        self.row_dim = self.page_size // self.storage_itemsize
        self.sample_row_count = math.ceil(self.sample_dim / self.row_dim)
        self.storage_sample_dim = self.sample_row_count * self.row_dim
        self.cache_dim = self.row_dim

        self.off = math.ceil(math.ceil(off / page_size) / num_ssd)
        self.num_ele = num_ele
        self.cache_size = cache_size
        self.cids_device = "cuda:" + str(ctrl_idx)

        self.iter_prefetch_depth = 4
        self.registered_split = int(os.environ.get("CIDS_REGISTERED_SPLIT", "1"))
        self.registered_try_window_size = int(os.environ.get(
            "CIDS_REGISTERED_TRY_WINDOW_SIZE", "2"))
        self.registered_enable_skip_front = os.environ.get(
            "CIDS_REGISTERED_ENABLE_SKIP_FRONT", "1") not in ("0", "", "false", "False")
        self.return_sample_ids = False

        self.GIDS_submit_time = 0.0
        self.GIDS_wait_time = 0.0

        self.GIDS_controller = BAM_Feature_Store.GIDS_Controllers()
        if ssd_list is None:
            self.ssd_list = [i for i in range(num_ssd)]
        else:
            self.ssd_list = ssd_list

        self.GIDS_controller.init_GIDS_controllers(num_ssd, 4096, 128, self.ssd_list)
        self.BAM_FS.init_controllers(
            self.GIDS_controller,
            page_size,
            self.off,
            cache_size,
            num_ele,
            num_ssd,
        )

    @classmethod
    def from_prepared_dataset(cls, prepared_root, **kwargs):
        # 直接从 prepared dataset 目录构建 CIDS。
        return cls(prepared_root=prepared_root, **kwargs)

    def store_tensor(self, in_ten, offset):
        # 把一个已经在 GPU 上的连续 tensor 写入 BaM 当前数组。
        num_e = len(in_ten)
        self.BAM_FS.store_tensor(in_ten.data_ptr(), num_e, offset)

    def enable_bam_policy_cache(self):
        # BaM policy 入口：
        # - policy table 按 sample 建，长度等于 num_samples
        # - 当 sample_row_count > 1 时，连续多个 row 共用同一条 sample score
        if self.meta is None:
            raise ValueError("enable_bam_policy_cache 需要 prepared dataset meta.json")
        num_policy_pages = int(self.meta["num_samples"])
        policy_group_pages = int(self.sample_row_count)
        self.return_sample_ids = True
        if policy_group_pages == 1:
            self.BAM_FS.enable_bam_policy_cache(num_policy_pages)
        else:
            self.BAM_FS.enable_bam_policy_cache_grouped(
                num_policy_pages,
                policy_group_pages,
            )
        return num_policy_pages

    def sync_bam_policy_scores(self, sample_ids, sample_scores):
        if torch.is_tensor(sample_ids):
            sample_ids = sample_ids.detach().cpu().view(-1).tolist()
        else:
            sample_ids = [int(sample_id) for sample_id in sample_ids]

        if torch.is_tensor(sample_scores):
            sample_scores = sample_scores.detach().cpu().view(-1).tolist()
        else:
            sample_scores = [float(score) for score in sample_scores]

        if len(sample_ids) != len(sample_scores):
            raise ValueError("sync_bam_policy_scores 需要等长的 sample_ids 和 sample_scores")
        if not sample_ids:
            return
        self.BAM_FS.update_bam_policy_scores(sample_ids, sample_scores)

    def sync_bam_policy_scores_device(self, policy_scores):
        # BaM policy 同步功能：
        # - 直接把 GPU 上的 sample/group 分数表 D2D 拷贝给 BaM
        # - 多 row 样本时，BaM 侧会用 group_pages 做地址到 sample 的映射
        if not torch.is_tensor(policy_scores):
            raise ValueError("sync_bam_policy_scores_device 需要 torch tensor")
        policy_scores = policy_scores.detach().to(device=self.cids_device, dtype=torch.float32).contiguous()
        self.BAM_FS.update_bam_policy_scores_device(
            policy_scores.data_ptr(),
            int(policy_scores.numel()),
        )

    def query_sample_residency(self, sample_ids):
        # 训练调度功能：
        # - sample 可能对应多个连续 row/page
        # - 只有这些 row/page 全部 resident，才把这个 sample 视为 resident
        if not torch.is_tensor(sample_ids):
            sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)
        sample_ids = sample_ids.detach().to(device=self.cids_device, dtype=torch.long).view(-1).contiguous()
        resident_mask = torch.zeros(
            sample_ids.numel(),
            dtype=torch.uint8,
            device=self.cids_device,
        )
        self.BAM_FS.query_sample_residency_device(
            sample_ids.data_ptr(),
            int(sample_ids.numel()),
            int(self.sample_row_count),
            resident_mask.data_ptr(),
        )
        return resident_mask.to(dtype=torch.bool).cpu()

    def _expand_sample_ids_to_row_indices(self, sample_ids):
        # 把 sample_id 展开成连续 row id：
        # 1 sample -> sample_row_count 个相邻 row。
        sample_ids = sample_ids.to(dtype=torch.long, device=self.cids_device).view(-1)
        row_offsets = torch.arange(
            self.sample_row_count,
            device=self.cids_device,
            dtype=torch.long,
        )
        row_indices = sample_ids.unsqueeze(1) * self.sample_row_count + row_offsets.unsqueeze(0)
        return row_indices.reshape(-1).contiguous()

    def _build_row_indices_from_batch(self, batch):
        # 为一个 batch 构造 row index tensor，并把它保留给整个 request 生命周期使用。
        sample_ids = self._sample_ids_from_batch(batch)
        return self._expand_sample_ids_to_row_indices(sample_ids)

    def load_prepared_images_to_bam(self, prepared_root=None, sample_id_offset=0):
        # 把 prepared dataset 的 images.bin 线性写入当前 BaM 数组。
        storage_array, meta = self._load_prepared_storage_array(prepared_root)
        num_samples = int(meta["num_samples"])
        tensor = torch.from_numpy(storage_array).to(self.cids_device).contiguous()
        offset_bytes = sample_id_offset * self.storage_sample_dim * tensor.element_size()
        self.store_tensor(tensor, offset_bytes)
        prepared_root = os.path.abspath(prepared_root if prepared_root is not None else self.prepared_root)
        return {
            "prepared_root": prepared_root,
            "num_samples": num_samples,
            "sample_id_offset": sample_id_offset,
            "offset_bytes": offset_bytes,
            "dtype": self.sample_dtype_name,
            "sample_row_count": self.sample_row_count,
            "row_dim": self.row_dim,
        }

    def _load_prepared_storage_array(self, prepared_root=None):
        # 读取 prepared dataset，并转换成适合当前 BaM row 路径写入的二维数组。
        # - uint8 prepared -> 直接按 uint8 写入 byte store
        # - float16/float32 prepared -> 维持现有 float32 写入路径
        if prepared_root is None:
            prepared_root = self.prepared_root
        if prepared_root is None:
            raise ValueError("_load_prepared_storage_array 需要 prepared_root")

        prepared_root = os.path.abspath(prepared_root)
        meta_path = os.path.join(prepared_root, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        sample_shape = tuple(meta["shape"])
        if sample_shape != self.sample_shape:
            raise ValueError(
                f"CIDS sample_shape={self.sample_shape} 与 prepared dataset shape={sample_shape} 不一致")

        dtype_name = meta["dtype"]
        np_dtype = _dtype_name_to_numpy_dtype(dtype_name)
        sample_dim = int(meta["sample_dim"])
        num_samples = int(meta["num_samples"])
        images_path = os.path.join(prepared_root, meta.get("images_file", "images.bin"))
        storage_sample_bytes = int(
            meta.get("sample_bytes", sample_dim * np.dtype(np_dtype).itemsize)
        )
        if storage_sample_bytes % np.dtype(np_dtype).itemsize != 0:
            raise ValueError(
                f"prepared sample_bytes 不能整除 itemsize: sample_bytes={storage_sample_bytes}, "
                f"itemsize={np.dtype(np_dtype).itemsize}"
            )
        storage_sample_dim = storage_sample_bytes // np.dtype(np_dtype).itemsize

        host_array = np.fromfile(images_path, dtype=np_dtype)
        expected_elems = num_samples * storage_sample_dim
        if host_array.size != expected_elems:
            raise ValueError(
                f"prepared images 元素数不匹配: got={host_array.size}, expected={expected_elems}")
        if storage_sample_dim < sample_dim:
            raise ValueError(
                f"prepared storage sample dim 过小: storage_sample_dim={storage_sample_dim}, "
                f"sample_dim={sample_dim}"
            )

        if self.sample_dtype_name == "uint8":
            if host_array.dtype != np.uint8:
                host_array = host_array.astype(np.uint8, copy=False)
            storage_dtype = np.uint8
            host_storage_array = host_array.reshape(num_samples, storage_sample_dim)
        else:
            host_storage_array = host_array.reshape(num_samples, storage_sample_dim)
            logical_array = _prepared_array_to_training_float32(
                host_storage_array[:, :sample_dim],
                dtype_name,
            )
            storage_dtype = np.float32
            if storage_sample_dim == sample_dim:
                host_storage_array = logical_array
            else:
                padded_storage = np.zeros(
                    (num_samples, storage_sample_dim),
                    dtype=storage_dtype,
                )
                padded_storage[:, :sample_dim] = logical_array
                host_storage_array = padded_storage

        if self.storage_sample_dim == storage_sample_dim:
            storage_array = host_storage_array.reshape(
                num_samples * self.sample_row_count,
                self.row_dim,
            )
        else:
            storage_array = np.zeros(
                (num_samples, self.storage_sample_dim),
                dtype=storage_dtype,
            )
            storage_array[:, :sample_dim] = host_storage_array[:, :sample_dim]
            storage_array = storage_array.reshape(num_samples * self.sample_row_count, self.row_dim)

        return storage_array, meta

    def load_combined_prepared_images_to_bam(self, prepared_roots):
        # 把多套 prepared dataset 在 Python 侧先拼成一块，再一次性写入 BaM。
        if not prepared_roots:
            raise ValueError("load_combined_prepared_images_to_bam 需要至少一个 prepared_root")

        storage_arrays = []
        infos = []
        sample_id_offset = 0

        for prepared_root in prepared_roots:
            storage_array, meta = self._load_prepared_storage_array(prepared_root)
            num_samples = int(meta["num_samples"])
            prepared_root = os.path.abspath(prepared_root)
            storage_arrays.append(storage_array)
            infos.append({
                "prepared_root": prepared_root,
                "num_samples": num_samples,
                "sample_id_offset": sample_id_offset,
                "dtype": self.sample_dtype_name,
                "sample_row_count": self.sample_row_count,
                "row_dim": self.row_dim,
            })
            sample_id_offset += num_samples

        combined_storage_array = np.concatenate(storage_arrays, axis=0)
        tensor = torch.from_numpy(combined_storage_array).to(self.cids_device).contiguous()
        self.store_tensor(tensor, 0)
        return infos

    def _sample_ids_from_batch(self, batch):
        # 约定 batch 的第一个字段是 sample_id tensor。
        if isinstance(batch, Mapping):
            if "sample_ids" not in batch:
                raise KeyError("CIDS batch 字典必须包含 sample_ids")
            sample_ids = batch["sample_ids"]
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 0:
                raise ValueError("CIDS batch 不能为空")
            sample_ids = batch[0]
        else:
            sample_ids = batch

        if not torch.is_tensor(sample_ids):
            sample_ids = torch.as_tensor(sample_ids)
        return sample_ids.to(self.cids_device)

    def _format_return_batch(self, batch, images):
        # 尽量保持 CNN 训练常见接口：(images, labels, ...)
        if self.return_sample_ids:
            if isinstance(batch, Mapping):
                result = dict(batch)
                result["images"] = images
                return result
            if isinstance(batch, tuple):
                labels = batch[1] if len(batch) > 1 else None
                return {"sample_ids": batch[0], "labels": labels, "images": images}
            if isinstance(batch, list):
                labels = batch[1] if len(batch) > 1 else None
                return {"sample_ids": batch[0], "labels": labels, "images": images}
            return {"sample_ids": batch, "images": images}

        if isinstance(batch, Mapping):
            result = dict(batch)
            result.pop("sample_ids", None)
            result["images"] = images
            return result

        if isinstance(batch, tuple):
            if len(batch) <= 1:
                return (images,)
            return (images, *batch[1:])

        if isinstance(batch, list):
            if len(batch) <= 1:
                return [images]
            return [images, *batch[1:]]

        return images

    def fetch_samples_submit_registered(self, batch, device):
        # 提交一个图片 batch 的异步 SSD->GPU 读取请求。
        s_time = time.time()
        row_indices = self._build_row_indices_from_batch(batch)
        request_id = self.BAM_FS.read_feature_submit_async_registered_rowctx(
            row_indices.data_ptr(),
            len(row_indices),
            self.row_dim,
            self.cache_dim,
            0,
        )
        self.GIDS_submit_time += time.time() - s_time
        return request_id, row_indices

    def fetch_samples_sync(self, batch, device):
        # 最原始同步路径：直接按展开后的 row index 读取整个 batch。
        s_time = time.time()
        sample_ids = self._sample_ids_from_batch(batch)
        row_indices = self._expand_sample_ids_to_row_indices(sample_ids)
        batch_size = len(sample_ids)
        images = torch.zeros(
            [batch_size * self.sample_row_count, self.row_dim],
            dtype=self.return_dtype,
            device=self.cids_device,
        ).contiguous()
        self.BAM_FS.read_feature(
            images.data_ptr(),
            row_indices.data_ptr(),
            len(row_indices),
            self.row_dim,
            self.cache_dim,
            0,
        )
        images = images.view(batch_size, self.storage_sample_dim)
        images = images[:, :self.sample_dim].contiguous()
        images = images.view(batch_size, *self.sample_shape)
        images = _images_to_training_tensor(images, self.sample_dtype_name)
        self.GIDS_wait_time += time.time() - s_time
        return self._format_return_batch(batch, images)

    def fetch_samples_get_registered(self, batch, device):
        # 获取一个已经 ready 的图片 batch，并 reshape 回 [B, C, H, W]。
        s_time = time.time()
        sample_ids = self._sample_ids_from_batch(batch)
        batch_size = len(sample_ids)
        images = torch.zeros(
            [batch_size * self.sample_row_count, self.row_dim],
            dtype=self.return_dtype,
            device=self.cids_device,
        ).contiguous()
        self.BAM_FS.read_feature_get_feature_light_registered_rowctx(
            images.data_ptr())
        images = images.view(batch_size, self.storage_sample_dim)
        images = images[:, :self.sample_dim].contiguous()
        images = images.view(batch_size, *self.sample_shape)
        images = _images_to_training_tensor(images, self.sample_dtype_name)
        self.GIDS_wait_time += time.time() - s_time
        return self._format_return_batch(batch, images)

    def service_registered_poll_compatible(self):
        # 单 outstanding 或兜底路径的阻塞轮询。
        s_time = time.time()
        request_id = self.BAM_FS.service_registered_poll_compatible()
        self.GIDS_wait_time += time.time() - s_time
        return request_id

    def service_registered_try_poll(self):
        # front request 的主轮询入口。
        s_time = time.time()
        request_id = self.BAM_FS.service_registered_try_poll()
        self.GIDS_wait_time += time.time() - s_time
        return request_id

    def service_registered_try_poll_window_skip_front(self, window_size):
        # 对 front 后面的 request 做轻量预热。
        s_time = time.time()
        request_id = self.BAM_FS.service_registered_try_poll_window_skip_front(
            window_size)
        self.GIDS_wait_time += time.time() - s_time
        return request_id

    def get_registered_outstanding_count(self):
        # 查看当前 registered outstanding request 数量。
        return self.BAM_FS.get_registered_outstanding_count()

    def get_registered_ready_front_request_id(self):
        # 返回当前 ready_front 对应的 request_id。
        return self.BAM_FS.get_registered_ready_front_request_id()

    def get_registered_request_id_at(self, offset):
        # 查看 outstanding 队列中某个 offset 的 request_id。
        return self.BAM_FS.get_registered_request_id_at(offset)

    def get_registered_request_state_at(self, offset):
        # 查看 outstanding 队列中某个 offset 的 request 状态。
        return self.BAM_FS.get_registered_request_state_at(offset)
