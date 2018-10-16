# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""

from collections import defaultdict, namedtuple
from functools import lru_cache
from numbers import Real
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor


def split_integers(total: int, fractions: Sequence[float]) -> List[int]:
    '''
    Splits an integer into smaller ones as given by fractions.
    May not be exact.
    '''
    if sum(fractions) != 1.0:
        # If these fractions don't exactly add to 1...
        raise ValueError(f"Expected fractions {fractions} to add to 1.0, got {sum(fractions)}")

    splits = [round(total * f) for f in fractions]

    sum_splits = sum(splits)
    if sum_splits != total:
        # If rounding errors brought us just off the total...
        difference = total - sum_splits
        splits[0] += difference
        # This handles cases where the difference is both positive and negative
        # TODO: Is there a better way to distribute the difference?

    return tuple(splits)


@lru_cache(maxsize=16)
def _make_mapping(device: torch.device) -> Tensor:
    return torch.as_tensor([[1, 0], [0, 1]], device=device, dtype=torch.long)


def expand_binary_class(output: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    '''
    Expand a binary classifier's single class from a scalar to a 2-tensor
    '''
    y_pred, y = output

    y_pred = _make_mapping(y_pred.device).index_select(0, y_pred.round().to(torch.long))

    return (y_pred, y.to(torch.long))


def to_singleton_row(y: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    '''
    Reshape a tensor to be a row of singletons, e.g. [[1], [2], [3], ...]
    '''
    return (y[0].reshape(-1, 1), y[1].reshape(-1, 1))

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.half,
    "float": torch.float,
    "double": torch.double,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "short": torch.short,
    "int": torch.int,
    "long": torch.long,
}  # type: Dict[str, torch.dtype]

Splits = namedtuple("Splits", ("full", "training", "validation", "testing"))
Metrics = namedtuple("Metrics", ("accuracy", "loss", "precision", "recall"))
NoneType = type(None)

JSONScalar = Union[NoneType, bool, str, Real]

NOT_BOT = 0
BOT = 1
