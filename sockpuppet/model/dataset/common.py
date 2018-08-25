import csv
import sys
from typing import Callable, Sequence, Tuple, Union
from numbers import Integral

import numpy
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

from sockpuppet.model.data import WordEmbeddings


def _to_int(i):
    if i in ("", "NULL"):
        return 0
    else:
        return int(i)


class TweetDataset(Dataset):
    pass


class TweetTensorDataset(Dataset):
    def __init__(
        self,
        data_source: TweetDataset,
        tokenizer: Callable[[str], Sequence[str]],
        embeddings: WordEmbeddings
    ):
        # TODO: Consider supporting reading directly from a file
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.data_source = data_source
        self.tensors = [None] * len(data_source)
        # NOTE: Each tensor might have a different shape, as each tensor represents a tweet

    @property
    def device(self):
        return self.embeddings.device

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index: Integral) -> Tensor:
        if self.tensors[index] is None:
            text = self.data_source[index].text
            tokens = self.tokenizer(text)
            self.tensors[index] = self.embeddings.encode(tokens)

        return self.tensors[index]
