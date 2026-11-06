from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenizedMultiLabelDataset(Dataset):
    def __init__(self, split_dir: str | Path, num_labels: int):
        self.split_dir = Path(split_dir)
        self.input_ids = np.load(self.split_dir / "input_ids.npy", mmap_mode="r")
        self.attention_mask = np.load(self.split_dir / "attention_mask.npy", mmap_mode="r")
        self.label_ptr = np.load(self.split_dir / "label_ptr.npy", mmap_mode="r")
        self.label_idx = np.load(self.split_dir / "label_idx.npy", mmap_mode="r")
        self.sample_ids = json.loads((self.split_dir / "sample_ids.json").read_text(encoding="utf-8"))
        self.num_labels = int(num_labels)
        n = len(self.sample_ids)
        if len(self.input_ids) != n or len(self.attention_mask) != n or len(self.label_ptr) != n + 1:
            raise RuntimeError("Token cache shape mismatch")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        a, b = int(self.label_ptr[idx]), int(self.label_ptr[idx + 1])
        labels = np.asarray(self.label_idx[a:b], dtype=np.int64)
        return {
            "input_ids": np.asarray(self.input_ids[idx], dtype=np.int64),
            "attention_mask": np.asarray(self.attention_mask[idx], dtype=np.int64),
            "label_indices": labels,
            "sample_id": self.sample_ids[idx],
        }


def make_collate_fn(num_labels: int):
    num_labels = int(num_labels)

    def collate(batch: List[dict]):
        input_ids = torch.tensor(np.stack([x["input_ids"] for x in batch]), dtype=torch.long)
        attention_mask = torch.tensor(np.stack([x["attention_mask"] for x in batch]), dtype=torch.long)
        targets = torch.zeros((len(batch), num_labels), dtype=torch.float32)
        label_lists = []
        for i, x in enumerate(batch):
            idx = torch.as_tensor(x["label_indices"], dtype=torch.long)
            if idx.numel():
                targets[i, idx] = 1.0
            label_lists.append(x["label_indices"].tolist())
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
            "label_indices": label_lists,
            "sample_ids": [x["sample_id"] for x in batch],
        }

    return collate
