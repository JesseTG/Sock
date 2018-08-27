from typing import Sequence, Union, Iterable
import csv

import numpy
import torch
from torch import Tensor
from torch.nn import Embedding
import pandas
from pandas import DataFrame

TORCH_INT_DTYPES = (torch.uint8, torch.int8, torch.short, torch.int, torch.long)


class WordEmbeddings:
    def __init__(self, path: Union[DataFrame, str], device="cpu"):
        if isinstance(path, str):
            with open(path, "r") as file:
                data = pandas.read_table(
                    file,
                    delim_whitespace=True,
                    header=None,
                    engine="c",
                    encoding="utf8",
                    na_filter=False,
                    memory_map=True,
                    quoting=csv.QUOTE_NONE
                )
        elif isinstance(path, DataFrame):
            data = path
        else:
            raise TypeError(f"Expected a str or DataFrame, got {path}")

        # self.words = data[0]
        # No need to keep around the wordlist separately, but if so we can just keep the dataframe

        self._dim = int(data.get_dtype_counts().float64)
        self.vectors = torch.as_tensor(data.iloc[:, 1:].values, dtype=torch.float, device=device)  # type: Tensor
        self.vectors.requires_grad_(False)
        # [all rows, second column:last column]

        # Pinning self.vectors does *not* improve encoding performance
        # torch.half isn't available for index_select, so we'll just use torch.float

        self.indices = {word: index for index, word in enumerate(data[0])}
        # note: must append a <unk> zero vector to embedding file
        # do so with python3 -c 'print("<unk>", *([0.0]*25))' >> the_data_file.txt

    def _get_word(self, index):
        return self.indices.get(index, 1)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def device(self) -> torch.device:
        return self.vectors.device

    @property
    def dim(self) -> int:
        return self._dim

    def __getitem__(self, index) -> Tensor:
        # Indexing uses the same underlying storage
        if isinstance(index, int):
            # If we're indexing by integer...
            return self.vectors[index]
        elif isinstance(index, str):
            # If we're trying to get a vector from a word...
            return self.vectors[self._get_word(index)]
        elif torch.is_tensor(index) and index.dim() == 0:
            # If this is a one-element tensor...
            return self.vectors[index]
        else:
            raise TypeError(f"Cannot index with a {type(index).__name__}")

    def encode(self, tokens: Sequence[str]) -> Tensor:
        if len(tokens) > 0:
            # If this is a non-empty sentence...
            return torch.as_tensor([self._get_word(t) for t in tokens], dtype=torch.long, device=self.device)
        else:
            return torch.as_tensor([1], dtype=torch.long, device=self.device)

    def to_layer(self) -> Embedding:
        return Embedding.from_pretrained(self.vectors)
