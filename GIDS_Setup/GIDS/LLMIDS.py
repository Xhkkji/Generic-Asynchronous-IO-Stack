import json
import os
from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .CIDS import (
        CIDS,
        _PrefetchingIter_async_registered_try_service_cids,
        _PrefetchingIter_sync_cids,
    )
except ImportError:  # pragma: no cover
    from CIDS import (  # type: ignore
        CIDS,
        _PrefetchingIter_async_registered_try_service_cids,
        _PrefetchingIter_sync_cids,
    )


class LLMIDS_DataLoader(DataLoader):
    # 面向 LLM/token 训练的 DataLoader 包装器：
    # 底层 dataset 只需要提供 sample_id，以及可选的 sequence length。
    def __init__(self, *args, LLMIDS_Loader=None, prefetch_depth=None, io_mode="registered",
                 registered_split=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.LLMIDS_Loader = LLMIDS_Loader
        self.prefetch_depth = int(prefetch_depth or getattr(
            self.LLMIDS_Loader, "iter_prefetch_depth", 1))
        self.io_mode = io_mode
        self.registered_split = int(registered_split or getattr(
            self.LLMIDS_Loader, "registered_split", 1))

    def __iter__(self):
        if self.io_mode == "sync":
            return _PrefetchingIter_sync_cids(
                super().__iter__(),
                self.LLMIDS_Loader,
            )
        if self.io_mode != "registered":
            raise ValueError(f"不支持的 LLMIDS io_mode: {self.io_mode}")
        return _PrefetchingIter_async_registered_try_service_cids(
            self,
            super().__iter__(),
            self.LLMIDS_Loader,
            prefetch_depth=self.prefetch_depth,
            registered_split=self.registered_split,
        )


class LLMIDSSampleDataset(Dataset):
    # 最小 LLM sample 索引数据集：
    # 每个样本只返回 sample_id，以及可选的有效 token 长度。
    def __init__(self, sample_ids, lengths=None):
        if lengths is not None and len(sample_ids) != len(lengths):
            raise ValueError("sample_ids 和 lengths 长度必须一致")
        self.sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)
        self.lengths = None if lengths is None else torch.as_tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        item = {
            "sample_ids": self.sample_ids[index],
        }
        if self.lengths is not None:
            item["lengths"] = self.lengths[index]
        return item


class LLMIDSPreparedDataset(LLMIDSSampleDataset):
    # 直接读取固定长度 token prepared dataset 目录：
    # 自动加载 meta.json 和可选的 lengths.npy，返回 sample_id + lengths。
    def __init__(self, prepared_root, start_sample_id=0):
        meta_path = os.path.join(prepared_root, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"未找到 LLMIDS meta.json: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        num_samples = int(meta["num_samples"])
        sample_ids = list(range(start_sample_id, start_sample_id + num_samples))

        lengths = None
        lengths_file = meta.get("lengths_file")
        if lengths_file is not None:
            lengths_path = os.path.join(prepared_root, lengths_file)
            if not os.path.exists(lengths_path):
                raise FileNotFoundError(f"未找到 LLMIDS lengths.npy: {lengths_path}")
            lengths = np.load(lengths_path)

        super().__init__(sample_ids, lengths=lengths)
        self.prepared_root = prepared_root
        self.meta = meta
        self.storage_sequence_length = int(meta["shape"][0])
        self.sequence_length = int(meta.get("sequence_length", self.storage_sequence_length))
        self.pad_token_id = int(meta.get("pad_token_id", 0))
        self.vocab_size = meta.get("vocab_size")


class LLMIDS(CIDS):
    # 面向固定长度 token block 的最小 LLM 读取骨架：
    # 每个 sample_id 对应一个 [seq_len] 的 int64 token 序列。
    def __init__(
        self,
        sample_shape=None,
        sequence_length=None,
        page_size=4096,
        off=0,
        num_ele=300 * 1000 * 1000 * 1024,
        num_ssd=1,
        ssd_list=None,
        cache_size=10,
        ctrl_idx=0,
        prepared_root=None,
        meta_path=None,
        pad_token_id=None,
        ignore_index=-100,
    ):
        if sample_shape is None and sequence_length is not None:
            sample_shape = (int(sequence_length),)

        super().__init__(
            sample_shape=sample_shape,
            page_size=page_size,
            off=off,
            num_ele=num_ele,
            num_ssd=num_ssd,
            ssd_list=ssd_list,
            cache_size=cache_size,
            ctrl_idx=ctrl_idx,
            long_type=True,
            prepared_root=prepared_root,
            meta_path=meta_path,
            dtype_name="int64",
        )
        if len(self.sample_shape) != 1:
            raise ValueError("LLMIDS 只支持一维固定长度 token 序列")
        if self.sample_dtype_name != "int64":
            raise ValueError("LLMIDS 当前只支持 int64 token 存储")

        self.storage_sequence_length = int(self.sample_shape[0])
        self.sequence_length = int(
            self.meta.get("sequence_length", self.storage_sequence_length)
        ) if self.meta else self.storage_sequence_length
        pad_token = self.meta.get("pad_token_id", pad_token_id) if self.meta else pad_token_id
        self.pad_token_id = None if pad_token is None else int(pad_token)
        self.ignore_index = int(ignore_index)
        self.default_return_labels = bool(
            self.meta.get("return_labels", True)
        ) if self.meta else True
        self.tokens_path = None
        self.lengths_path = None
        if self.meta is not None and self.prepared_root is not None:
            self.tokens_path = os.path.join(
                self.prepared_root, self.meta.get("tokens_file", "tokens.bin"))
            lengths_file = self.meta.get("lengths_file")
            if lengths_file is not None:
                self.lengths_path = os.path.join(self.prepared_root, lengths_file)

    def _load_prepared_storage_array(self, prepared_root=None):
        # 读取 prepared token dataset，并转换成适合当前 BaM row 路径写入的二维数组。
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
                f"LLMIDS sample_shape={self.sample_shape} 与 prepared dataset shape={sample_shape} 不一致")

        dtype_name = meta.get("dtype", "int64")
        if dtype_name != "int64":
            raise ValueError(f"LLMIDS 只支持 int64 token prepared dataset，当前为 {dtype_name}")

        sample_dim = int(meta["sample_dim"])
        num_samples = int(meta["num_samples"])
        tokens_path = os.path.join(prepared_root, meta.get("tokens_file", "tokens.bin"))
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"未找到 LLMIDS tokens.bin: {tokens_path}")
        host_array = np.fromfile(tokens_path, dtype=np.int64)
        expected_elems = num_samples * sample_dim
        if host_array.size != expected_elems:
            raise ValueError(
                f"prepared tokens 元素数不匹配: got={host_array.size}, expected={expected_elems}")

        host_array = host_array.reshape(num_samples, sample_dim)
        if self.storage_sample_dim == sample_dim:
            storage_array = host_array.reshape(num_samples * self.sample_row_count, self.row_dim)
        else:
            storage_array = np.zeros(
                (num_samples, self.storage_sample_dim),
                dtype=np.int64,
            )
            storage_array[:, :sample_dim] = host_array
            storage_array = storage_array.reshape(num_samples * self.sample_row_count, self.row_dim)
        return storage_array, meta

    def load_prepared_tokens_to_bam(self, prepared_root=None, sample_id_offset=0):
        # 把 prepared dataset 的 tokens.bin 线性写入当前 BaM 数组。
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
            "sequence_length": self.sequence_length,
        }

    def load_combined_prepared_tokens_to_bam(self, prepared_roots):
        # 把多套 prepared token dataset 在 Python 侧先拼成一块，再一次性写入 BaM。
        if not prepared_roots:
            raise ValueError("load_combined_prepared_tokens_to_bam 需要至少一个 prepared_root")

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
                "sequence_length": self.sequence_length,
            })
            sample_id_offset += num_samples

        combined_storage_array = np.concatenate(storage_arrays, axis=0)
        tensor = torch.from_numpy(combined_storage_array).to(self.cids_device).contiguous()
        self.store_tensor(tensor, 0)
        return infos

    def _lengths_from_batch(self, batch):
        if isinstance(batch, Mapping):
            lengths = batch.get("lengths")
            if lengths is None:
                return None
            if not torch.is_tensor(lengths):
                lengths = torch.as_tensor(lengths)
            return lengths.to(self.cids_device)
        return None

    def _attention_mask_from_batch_or_tokens(self, batch, input_ids):
        if isinstance(batch, Mapping) and "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
            if not torch.is_tensor(attention_mask):
                attention_mask = torch.as_tensor(attention_mask)
            return attention_mask.to(self.cids_device)

        lengths = self._lengths_from_batch(batch)
        if lengths is not None:
            positions = torch.arange(
                self.sequence_length,
                device=input_ids.device,
                dtype=lengths.dtype,
            ).unsqueeze(0)
            return (positions < lengths.view(-1, 1)).to(torch.long)

        if self.pad_token_id is None:
            return torch.ones_like(input_ids, dtype=torch.long)
        return input_ids.ne(self.pad_token_id).to(torch.long)

    def _labels_from_batch_or_tokens(self, batch, input_ids, attention_mask):
        if isinstance(batch, Mapping) and "labels" in batch:
            labels = batch["labels"]
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels)
            return labels.to(self.cids_device)

        if not self.default_return_labels:
            return None

        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, self.ignore_index)
        return labels

    def _format_return_batch(self, batch, input_ids):
        attention_mask = self._attention_mask_from_batch_or_tokens(batch, input_ids)
        labels = self._labels_from_batch_or_tokens(batch, input_ids, attention_mask)

        if isinstance(batch, Mapping):
            result = dict(batch)
            result.pop("sample_ids", None)
            result["input_ids"] = input_ids
            result["attention_mask"] = attention_mask
            if labels is not None:
                result["labels"] = labels
            return result

        result = {"input_ids": input_ids}
        result["attention_mask"] = attention_mask
        if labels is not None:
            result["labels"] = labels
        return result

    def fetch_samples_sync(self, batch, device):
        # 最原始同步路径：直接按 sample_id 读取整个 token batch。
        sample_ids = self._sample_ids_from_batch(batch)
        row_indices = self._expand_sample_ids_to_row_indices(sample_ids)
        batch_size = len(sample_ids)
        input_ids = torch.zeros(
            [batch_size * self.sample_row_count, self.row_dim],
            dtype=self.return_dtype,
            device=self.cids_device,
        ).contiguous()
        self.BAM_FS.read_feature(
            input_ids.data_ptr(),
            row_indices.data_ptr(),
            len(row_indices),
            self.row_dim,
            self.cache_dim,
            0,
        )
        input_ids = input_ids.view(batch_size, self.storage_sample_dim)
        input_ids = input_ids[:, :self.storage_sequence_length].contiguous()
        return self._format_return_batch(batch, input_ids)

    def fetch_samples_get_registered(self, batch, device):
        # 获取一个已经 ready 的 token batch，并 reshape 回 [B, seq_len]。
        sample_ids = self._sample_ids_from_batch(batch)
        batch_size = len(sample_ids)
        input_ids = torch.zeros(
            [batch_size * self.sample_row_count, self.row_dim],
            dtype=self.return_dtype,
            device=self.cids_device,
        ).contiguous()
        self.BAM_FS.read_feature_get_feature_light_registered_rowctx(
            input_ids.data_ptr())
        input_ids = input_ids.view(batch_size, self.storage_sample_dim)
        input_ids = input_ids[:, :self.storage_sequence_length].contiguous()
        return self._format_return_batch(batch, input_ids)
