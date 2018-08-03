import csv
import sys
from typing import Callable, Sequence, Tuple, Union

import numpy
import pandas
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

from sockpuppet.model.embedding import WordEmbeddings


def _to_int(i):
    if i in ("", "NULL"):
        return 0
    else:
        return int(i)

TWEET_COLUMN_TYPES = {
    "text": str,
    "retweet_count": int,
    "reply_count": int,
    "favorite_count": int,
    "num_hashtags": int,
    "num_urls": int,
    "num_mentions": int,
}

USER_COLUMN_TYPES = {
    "statuses_count": int,
    "followers_count": int,
    "friends_count": int,
    "favourites_count": int,
    "listed_count": int,
    "default_profile": bool,
    "profile_use_background_image": bool,
    "protected": bool,
    "verified": bool
}
# TODO: Go to the official Twitter docs to see what the exact types of these fields should be


class CresciTweetDataset(Dataset):
    def __init__(self, path: str):
        self.data = pandas.read_csv(
            path,
            dtype=TWEET_COLUMN_TYPES,
            engine="c",
            usecols=tuple(TWEET_COLUMN_TYPES.keys()),
            encoding="utf8",
            header=0,
            memory_map=True,
            na_filter=False,
            doublequote=True,
            error_bad_lines=False
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data.loc[index]


class CresciTensorTweetDataset(Dataset):
    def __init__(
        self,
        data_source: CresciTweetDataset,
        tokenizer: Callable[[str], Sequence[str]],
        embeddings: WordEmbeddings,
        device: Union[str, torch.device]="cpu"
    ):
        # TODO: Consider supporting reading directly from a file
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.data_source = data_source
        self.tensors = [None] * len(data_source)
        self.device = device
        # NOTE: Each tensor might have a different shape, as each tensor represents a tweet

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index: int) -> Tensor:
        if self.tensors[index] is None:
            text = self.data_source[index].text
            tokens = self.tokenizer(text)
            tensor = self.embeddings.encode(tokens)
            self.tensors[index] = tensor.to(self.device)

        return self.tensors[index]


class CresciUserDataset(Dataset):
    def __init__(self, path: str):
        self.data = pandas.read_csv(
            path,
            dtype=USER_COLUMN_TYPES,
            engine="c",
            encoding="utf-8",
            memory_map=True,
            na_filter=False,
            false_values=[""],
            true_values=["1"]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data.loc[index]
