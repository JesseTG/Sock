import pytest
import torch
from sockpuppet.model.dataset.five38 import Five38TweetDataset, Five38TweetTensorDataset
from sockpuppet.model.dataset.twitter_tokenize import tokenize
from sockpuppet.model.embedding import WordEmbeddings

from .marks import *


def test_538_tweet_row_loaded(five38_tweets: Five38TweetDataset):
    assert five38_tweets[3].text == "Amen! #blacklivesmatter https://t.co/wGffaOqgzl"


def test_538_all_tweets_loaded(five38_tweets: Five38TweetDataset):
    assert len(five38_tweets) == 203451


def test_538_loads_multiline_tweet(five38_tweets: Five38TweetDataset):
    assert five38_tweets[4].text == "RT @NahBabyNah: Twitchy: Chuck Todd caught out there shilling for Hillary Clinton\r\nThe post BUSTED: Adam Baldwi... https://t.co/ay28pMpDw6 #…"


@modes("cpu", "cuda")
def test_538_tweet_tensor_device_is_correct(device: torch.device, five38_tweets_tensors: Five38TweetTensorDataset):
    assert five38_tweets_tensors.device == device


@modes("cpu", "cuda")
@pytest.mark.benchmark(group="538_tweet_tensor_cached_access", warmup=True)
def test_bench_538_tweet_tensor_cached_access(benchmark, five38_tweets_tensors: Five38TweetTensorDataset):
    result = benchmark(five38_tweets_tensors.__getitem__, 0)
    assert result is not None


@modes("cpu", "cuda")
@pytest.mark.benchmark(group="538_tweet_tensor_fresh_access")
def test_bench_538_tweet_tensor_fresh_access(benchmark, five38_tweets: Five38TweetDataset, glove_embedding: WordEmbeddings, device: torch.device):
    # TODO: Make this benchmark run a different set of indices for each iteration
    dataset = Five38TweetTensorDataset(
        data_source=five38_tweets,
        embeddings=glove_embedding,
        tokenizer=tokenize
    )

    def access():
        return [dataset[i] for i in range(1000)]

    result = benchmark.pedantic(access)

    assert result is not None


@modes("cpu", "cuda")
def test_538_tweet_tensor_dataset_is_created(five38_tweets_tensors: Five38TweetTensorDataset):
    assert five38_tweets_tensors is not None


@modes("cpu", "cuda")
def test_538_tweet_tensor_dataset_loads_tensors(five38_tweets_tensors: Five38TweetTensorDataset):
    assert torch.is_tensor(five38_tweets_tensors[0])


@modes("cpu", "cuda")
def test_538_tweet_tensor_encodes_all_words(five38_tweets_tensors: Five38TweetTensorDataset):
    assert len(five38_tweets_tensors[0]) == 7


@modes("cpu", "cuda")
def test_538_tweet_tensor_dataset_returns_word_indices(five38_tweets_tensors: Five38TweetTensorDataset):
    assert five38_tweets_tensors[0].dtype == torch.long
