"""PyTorch dataset classes with sequence packing and deterministic resumable sampling."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ankalm.core.constants import SPECIAL_TOKEN_IDS


class PackedTokenDataset(Dataset):
    """
    High-throughput sequence-packed causal language modeling dataset.
    Concatenates tokenized documents separated by <|bos|> and <|eos|> into contiguous
    chunks of length (seq_len + 1), producing (input_ids, labels) where labels are shifted by 1.
    """

    def __init__(
        self,
        token_ids: List[int] | torch.Tensor,
        seq_len: int = 256,
        stride: Optional[int] = None,
    ) -> None:
        super().__init__()
        if isinstance(token_ids, torch.Tensor):
            self.data = token_ids.to(dtype=torch.long)
        else:
            self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride or seq_len

        total_len = self.data.numel()
        if total_len <= seq_len + 1:
            # Pad or tile minimal sequence so small smoke tests always have at least 1 sample
            repeats = ((seq_len + 2) // max(total_len, 1)) + 2
            self.data = self.data.repeat(repeats)
            total_len = self.data.numel()

        self.starts = list(range(0, total_len - seq_len - 1, self.stride))
        if not self.starts:
            self.starts = [0]

    @property
    def total_tokens(self) -> int:
        return int(self.data.numel())

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index % len(self.starts)]
        chunk = self.data[start : start + self.seq_len + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: Path,
        encode_fn: Callable[[str], List[int]],
        seq_len: int = 256,
        stride: Optional[int] = None,
        cache_pt_path: Optional[Path] = None,
        repeat_epochs: int = 1,
    ) -> "PackedTokenDataset":
        bos_id = SPECIAL_TOKEN_IDS["BOS"]
        eos_id = SPECIAL_TOKEN_IDS["EOS"]
        packed: List[int] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                if not text:
                    continue
                ids = encode_fn(text)
                packed.append(bos_id)
                packed.extend(ids)
                packed.append(eos_id)

        if repeat_epochs > 1:
            packed = packed * repeat_epochs

        tensor_data = torch.tensor(packed, dtype=torch.long)
        if cache_pt_path is not None:
            cache_pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor_data, cache_pt_path)
        return cls(tensor_data, seq_len=seq_len, stride=stride)


class CausalTextDataset(Dataset):
    """Document-level causal dataset with padding and ignore_index=-100 on pad tokens."""

    def __init__(
        self,
        samples: List[List[int]],
        seq_len: int = 256,
        pad_id: int = SPECIAL_TOKEN_IDS["PAD"],
    ) -> None:
        super().__init__()
        self.samples = samples
        self.seq_len = seq_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.samples[idx][: self.seq_len + 1]
        if len(tokens) < self.seq_len + 1:
            pad_len = (self.seq_len + 1) - len(tokens)
            tokens = tokens + [self.pad_id] * pad_len

        t = torch.tensor(tokens, dtype=torch.long)
        x = t[:-1].clone()
        y = t[1:].clone()
        y[y == self.pad_id] = -100
        return x, y
