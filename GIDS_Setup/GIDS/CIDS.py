import math
import os
import time
import json
from collections.abc import Mapping

import torch
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


class _PrefetchingIter_async_registered_try_service_cids(object):
    # CIDS 的最小 registered try-service 迭代器：
    # 按 sample_id 提交图片读取，请求完成后再把图片 batch 返回给训练。
    def __init__(self, dataloader, dataloader_it, CIDS_Loader=None, prefetch_depth=1):
        self.dataloader = dataloader
        self.dataloader_it = dataloader_it
        self.CIDS_Loader = CIDS_Loader

        self.prefetch_queue = []
        self.exhausted = False
        self.prefetch_depth = max(1, int(prefetch_depth))
        self.enable_skip_front = bool(getattr(
            self.CIDS_Loader, "registered_enable_skip_front", True))
        self.try_window_size = max(1, int(getattr(
            self.CIDS_Loader, "registered_try_window_size", 1)))
        self.debug = os.environ.get(
            "CIDS_DEBUG", "0") not in ("0", "", "false", "False")

        self._warmup_prefetch_queue()

    def _debug_log(self, message):
        # 统一调试输出入口，默认关闭。
        if self.debug:
            print(f"[CIDS] {message}", flush=True)

    def _warmup_prefetch_queue(self):
        # 初始化时先把预取队列填到目标深度。
        while len(self.prefetch_queue) < self.prefetch_depth and not self.exhausted:
            if not self._submit_prefetch():
                break

    def _submit_prefetch(self):
        # 从底层 dataloader 取一个 batch，并提交对应图片读取请求。
        if self.exhausted:
            return False

        try:
            next_batch = next(self.dataloader_it)
        except StopIteration:
            self.exhausted = True
            return False

        request_id = self.CIDS_Loader.fetch_samples_submit_registered(
            next_batch, self.CIDS_Loader.cids_device)
        self.prefetch_queue.append({
            "batch": next_batch,
            "request_id": request_id,
        })
        self._debug_log(
            f"提交完成 request_id={request_id}, queue_len={len(self.prefetch_queue)}")
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

    def _poll_front_prefetch(self):
        # 强轮询当前 front request，直到它 ready。
        if not self.prefetch_queue:
            return

        expected = self.prefetch_queue[0]["request_id"]
        ready_front = self.CIDS_Loader.get_registered_ready_front_request_id()
        if ready_front == expected:
            return

        try_loops = 0
        while True:
            outstanding = self.CIDS_Loader.get_registered_outstanding_count()
            if outstanding <= 1:
                request_id = self.CIDS_Loader.service_registered_poll_compatible()
            else:
                request_id = self.CIDS_Loader.service_registered_try_poll()
                try_loops += 1
                if try_loops % 8 == 0:
                    self._service_prefetch_nonblocking()

            if request_id == expected:
                return

    def __iter__(self):
        return self

    def __next__(self):
        # 返回一个完整的 CNN batch：(images, labels, ...)
        if not self.prefetch_queue and self.exhausted:
            raise StopIteration

        self._service_prefetch_nonblocking()
        self._poll_front_prefetch()

        next_item = self.prefetch_queue.pop(0)
        batch = self.CIDS_Loader.fetch_samples_get_registered(
            next_item["batch"], self.CIDS_Loader.cids_device)

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

    def __iter__(self):
        return self

    def __next__(self):
        next_batch = next(self.dataloader_it)
        return self.CIDS_Loader.fetch_samples_sync(
            next_batch, self.CIDS_Loader.cids_device)


class CIDS_DataLoader(DataLoader):
    # 面向 CNN/image 训练的 DataLoader 包装器：
    # 底层 dataset 只需要提供 sample_id 和 label。
    def __init__(self, *args, CIDS_Loader=None, prefetch_depth=None, io_mode="registered",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.CIDS_Loader = CIDS_Loader
        self.prefetch_depth = int(prefetch_depth or getattr(
            self.CIDS_Loader, "iter_prefetch_depth", 1))
        self.io_mode = io_mode

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
        else:
            self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store_float()
            # 当前 CIDS 先统一按 float32 存入/读取 BaM 数组，减少和现有
            # BAM_Feature_Store_float 路径的类型错位。
            self.return_dtype = torch.float32

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
        # 当前先复用 BaM 现有的 float row 路径：
        # 1 张图片不会直接当成 1 行，而是拆成若干个连续 row。
        self.storage_itemsize = 4
        self.row_dim = self.page_size // self.storage_itemsize
        self.sample_row_count = math.ceil(self.sample_dim / self.row_dim)
        self.storage_sample_dim = self.sample_row_count * self.row_dim
        self.cache_dim = self.row_dim

        self.off = math.ceil(math.ceil(off / page_size) / num_ssd)
        self.num_ele = num_ele
        self.cache_size = cache_size
        self.cids_device = "cuda:" + str(ctrl_idx)

        self.iter_prefetch_depth = 4
        self.registered_try_window_size = int(os.environ.get(
            "CIDS_REGISTERED_TRY_WINDOW_SIZE", "2"))
        self.registered_enable_skip_front = os.environ.get(
            "CIDS_REGISTERED_ENABLE_SKIP_FRONT", "1") not in ("0", "", "false", "False")

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
            "dtype": "float32",
            "sample_row_count": self.sample_row_count,
            "row_dim": self.row_dim,
        }

    def _load_prepared_storage_array(self, prepared_root=None):
        # 读取 prepared dataset，并转换成适合 BaM row 写入的二维 float32 数组。
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

        host_array = np.fromfile(images_path, dtype=np_dtype)
        expected_elems = num_samples * sample_dim
        if host_array.size != expected_elems:
            raise ValueError(
                f"prepared images 元素数不匹配: got={host_array.size}, expected={expected_elems}")

        if host_array.dtype != np.float32:
            host_array = host_array.astype(np.float32, copy=False)

        host_array = host_array.reshape(num_samples, sample_dim)
        if self.storage_sample_dim == sample_dim:
            storage_array = host_array.reshape(num_samples * self.sample_row_count, self.row_dim)
        else:
            storage_array = np.zeros(
                (num_samples, self.storage_sample_dim),
                dtype=np.float32,
            )
            storage_array[:, :sample_dim] = host_array
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
                "dtype": "float32",
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
        sample_ids = self._sample_ids_from_batch(batch)
        row_indices = self._expand_sample_ids_to_row_indices(sample_ids)
        request_id = self.BAM_FS.read_feature_submit_async_registered_rowctx(
            row_indices.data_ptr(),
            len(row_indices),
            self.row_dim,
            self.cache_dim,
            0,
        )
        self.GIDS_submit_time += time.time() - s_time
        return request_id

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
