import pytest
import torch
from sockpuppet.model.dataset.nbc import NbcTweetDataset, NbcTweetTensorDataset
from sockpuppet.model.dataset.twitter_tokenize import tokenize
from sockpuppet.model.embedding import WordEmbeddings

from .marks import *


def test_nbc_tweet_row_loaded(nbc_tweets: NbcTweetDataset):
    assert nbc_tweets[3].text == "Amen! #blacklivesmatter https://t.co/wGffaOqgzl"


def test_nbc_all_tweets_loaded(nbc_tweets: NbcTweetDataset):
    assert len(nbc_tweets) == 203451


def test_nbc_loads_multiline_tweet(nbc_tweets: NbcTweetDataset):
    assert nbc_tweets[4].text == "RT @NahBabyNah: Twitchy: Chuck Todd caught out there shilling for Hillary Clinton\r\nThe post BUSTED: Adam Baldwi... https://t.co/ay28pMpDw6 #…"


@modes("cpu", "cuda")
def test_nbc_tweet_tensor_device_is_correct(device: torch.device, nbc_tweets_tensors: NbcTweetTensorDataset):
    assert nbc_tweets_tensors.device == device


@modes("cpu", "cuda")
@pytest.mark.benchmark(group="nbc_tweet_tensor_cached_access", warmup=True)
def test_bench_nbc_tweet_tensor_cached_access(benchmark, nbc_tweets_tensors: NbcTweetTensorDataset):
    result = benchmark(nbc_tweets_tensors.__getitem__, 0)
    assert result is not None


@modes("cpu", "cuda")
@pytest.mark.benchmark(group="nbc_tweet_tensor_fresh_access")
def test_bench_nbc_tweet_tensor_fresh_access(benchmark, nbc_tweets: NbcTweetDataset, glove_embedding: WordEmbeddings, device: torch.device):
    # TODO: Make this benchmark run a different set of indices for each iteration
    dataset = NbcTweetTensorDataset(
        data_source=nbc_tweets,
        embeddings=glove_embedding,
        tokenizer=tokenize
    )

    def access():
        return [dataset[i] for i in range(1000)]

    result = benchmark.pedantic(access)

    assert result is not None


@modes("cpu", "cuda")
def test_nbc_tweet_tensor_dataset_is_created(nbc_tweets_tensors: NbcTweetTensorDataset):
    assert nbc_tweets_tensors is not None


@modes("cpu", "cuda")
def test_nbc_tweet_tensor_dataset_loads_tensors(nbc_tweets_tensors: NbcTweetTensorDataset):
    assert torch.is_tensor(nbc_tweets_tensors[0])


@modes("cpu", "cuda")
def test_nbc_tweet_tensor_encodes_all_words(nbc_tweets_tensors: NbcTweetTensorDataset):
    assert len(nbc_tweets_tensors[0]) == 7


@modes("cpu", "cuda")
def test_nbc_tweet_tensor_dataset_returns_word_indices(nbc_tweets_tensors: NbcTweetTensorDataset):
    assert nbc_tweets_tensors[0].dtype == torch.long
