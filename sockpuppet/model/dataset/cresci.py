import csv
import sys
from typing import Callable, Sequence, Tuple, Union
from numbers import Integral

import numpy
import pandas
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

from sockpuppet.model.data import WordEmbeddings
from .common import TweetDataset, TweetTensorDataset, _to_int

TWEET_COLUMN_TYPES = {
    "text": str,
    "retweet_count": int,
    "reply_count": int,
    "favorite_count": int,
    "num_hashtags": int,
    "num_urls": int,
    "num_mentions": int,
}

TWEET_COLUMN_NAMES = tuple(TWEET_COLUMN_TYPES.keys())

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


class CresciTweetDataset(TweetDataset):
    def __init__(self, path: str):
        self.data = pandas.read_csv(
            path,
            dtype=TWEET_COLUMN_TYPES,
            engine="c",
            usecols=TWEET_COLUMN_NAMES,
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
        if torch.is_tensor(index):
            index = index.item()

        return self.data.loc[index]


class CresciTensorTweetDataset(TweetTensorDataset):
    pass


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

    def __getitem__(self, index: Integral):
        return self.data.loc[index]
