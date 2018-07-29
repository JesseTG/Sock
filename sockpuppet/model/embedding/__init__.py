from typing import Sequence, Union, Iterable
import csv

import numpy
import torch
from torch import Tensor
from torch.nn import Embedding
import pandas

TORCH_INT_DTYPES = (torch.uint8, torch.int8, torch.short, torch.int, torch.long)


class WordEmbeddings:
    def __init__(self, path: str, dim: int):
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
        self.vectors = torch.from_numpy(data.iloc[:, 1:dim + 1].values).type(torch.float)  # type: Tensor
        self.vectors.requires_grad_(False)
        # [all rows, second column:last column]

        self.indices = {word: index for index, word in enumerate(self.words)}
        # note: must append a <unk> zero vector to embedding file
        # do so with python3 -c 'print("<unk>", *([0.0]*25))' >> the_data_file.txt

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index) -> Tensor:
        if isinstance(index, int):
            # If we're indexing by integer...
            return self.vectors[index]
        elif isinstance(index, str):
            # If we're trying to get a vector from a word...
            return self.vectors[self.indices.get(index, len(self) - 1)]
        elif torch.is_tensor(index) and index.dim() == 0:
            # If this is a one-element tensor...
            return self.vectors[index.item()]
        else:
            raise TypeError(f"Cannot index with a {type(index).__name__}")

    def encode(self, tokens: Sequence[str]) -> Tensor:
        return torch.LongTensor([self.indices.get(t, len(self) - 1) for t in tokens])
        # Encodings should always be on the CPU, as the dictionary is

    def to_layer(self) -> Embedding:
        return Embedding.from_pretrained(self.vectors)

    def embed(self, encoding) -> Tensor:
        if len(encoding) == 0:
            # If we're embedding an empty document...
            return torch.zeros([1, self.dim], dtype=torch.float)
            # ...return the zero vector
        elif torch.is_tensor(encoding) and encoding.dtype in TORCH_INT_DTYPES:
            # Else if this is a tensor of indices...
            return self.vectors.index_select(0, encoding)
            # TODO: Can integer-based tensors besides LongTensor be used for indices?
        elif isinstance(encoding, list) or isinstance(encoding, tuple):
            # Else if this is a standard Python sequence...
            if isinstance(encoding[0], int):
                # ...of word indices...
                return self.vectors.index_select(0, torch.LongTensor(encoding))
            elif isinstance(encoding[0], str):
                # ...of words...
                return self.vectors.index_select(0, self.encode(encoding))
            else:
                raise TypeError(f"{type(encoding).__name__} must be made of ints or strings")
        else:
            raise TypeError(f"Don't know how to embed a {type(encoding).__name__}")
