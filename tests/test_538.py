import pytest
import torch

from sockpuppet.model.data import WordEmbeddings, tokenize
from sockpuppet.model.dataset import Five38TweetDataset, Five38TweetTensorDataset

from .marks import *


def test_538_tweet_row_loaded(five38_tweets: Five38TweetDataset):
    assert five38_tweets[5].text == 'Dan Bongino: "Nobody trolls liberals better than Donald Trump." Exactly!  https://t.co/AigV93aC8J'


def test_538_all_tweets_loaded(five38_tweets: Five38TweetDataset):
    assert len(five38_tweets) == 2973379


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
    assert len(five38_tweets_tensors[0]) == 28


@modes("cpu", "cuda")
def test_538_tweet_tensor_dataset_returns_word_indices(five38_tweets_tensors: Five38TweetTensorDataset):
    assert five38_tweets_tensors[0].dtype == torch.long
