from typing import Any, Sequence, Tuple, TypeVar

import torch
from torch.utils.data.dataset import Dataset

T = TypeVar('T')
U = TypeVar('U')


class LabelDataset(Dataset):
    def __init__(self, data: Sequence[T], labels: Sequence[U]):
        self.data = data
        self.labels = labels

        if len(data) != len(labels):
            raise ValueError(f"data and labels must have the same length ({len(data)} vs {len(labels)})")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[T, U]:
        if torch.is_tensor(index):
            index = index.item()

        return (self.data[index], self.labels[index])


class SingleLabelDataset(Dataset):
    def __init__(self, data: Sequence[T], label: Any):
        self.data = data
        self.label = label

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[T, Any]:
        if torch.is_tensor(index):
            index = index.item()

        return (self.data[index], self.label)
