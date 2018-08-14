from typing import Sequence, Union, Iterable
import csv

import numpy
import torch
from torch import Tensor
from torch.nn import Embedding
import pandas

TORCH_INT_DTYPES = (torch.uint8, torch.int8, torch.short, torch.int, torch.long)


class WordEmbeddings:
    # TODO: Add support for cuda devices
    def __init__(self, path: str, dim: int, device="cpu"):
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

        self.dim = dim
        self.words = data[0].tolist()  # type: list
        self.vectors = torch.from_numpy(data.iloc[:, 1:dim + 1].values).type(torch.float).to(device)  # type: Tensor
        self.vectors.requires_grad_(False)
        # [all rows, second column:last column]

        self.indices = {word: index for index, word in enumerate(self.words)}
        # note: must append a <unk> zero vector to embedding file
        # do so with python3 -c 'print("<unk>", *([0.0]*25))' >> the_data_file.txt

    def _get_word(self, index):
        return self.indices.get(index, 1)

    def __len__(self) -> int:
        return len(self.words)

    @property
    def device(self) -> torch.device:
        return self.vectors.device

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
            return self.vectors[index.item()]
        else:
            raise TypeError(f"Cannot index with a {type(index).__name__}")

    def encode(self, tokens: Sequence[str]) -> Tensor:
        if len(tokens) > 0:
            # If this is a non-empty sentence...
            return torch.tensor([self._get_word(t) for t in tokens], dtype=torch.long, device=self.device)
        else:
            return torch.tensor([1], dtype=torch.long, device=self.device)
            # TODO: Dedicate a special token <empty> and an index for it

    def to_layer(self) -> Embedding:
        return Embedding.from_pretrained(self.vectors)

    def embed(self, encoding) -> Tensor:
        if len(encoding) == 0:
            # If we're embedding an empty document...
            return torch.zeros([1, self.dim], dtype=torch.float, device=self.device)
            # ...return the zero vector
        elif torch.is_tensor(encoding) and encoding.dtype in TORCH_INT_DTYPES:
            # Else if this is a tensor of indices...
            return self.vectors.index_select(0, encoding)
            # TODO: Can integer-based tensors besides LongTensor be used for indices?
        elif isinstance(encoding, Sequence):
            # Else if this is a standard Python sequence...
            if isinstance(encoding[0], int):
                # ...of word indices...
                return self.vectors.index_select(0, torch.tensor(encoding, dtype=torch.long, device=self.device))
            elif isinstance(encoding[0], str):
                # ...of words...
                return self.vectors.index_select(0, self.encode(encoding))
            else:
                raise TypeError(f"{type(encoding).__name__} must be made of ints or strings")
        else:
            raise TypeError(f"Don't know how to embed a {type(encoding).__name__}")
