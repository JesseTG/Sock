import pytest
import torch
from sockpuppet.model.dataset.cresci import CresciTweetDataset, CresciUserDataset, CresciTensorTweetDataset
from sockpuppet.model.data import tokenize, WordEmbeddings

from .marks import *


def test_cresci_tweet_row_loaded(cresci_genuine_accounts_tweets: CresciTweetDataset):
    assert cresci_genuine_accounts_tweets[5].text == "Don't. Ok? https://t.co/uTXrJ6zvdN"
    assert cresci_genuine_accounts_tweets[5].num_urls == 1


def test_cresci_all_tweets_loaded(cresci_genuine_accounts_tweets: CresciTweetDataset):
    assert len(cresci_genuine_accounts_tweets) == 2839361


def test_cresci_loads_empty_tweet(cresci_genuine_accounts_tweets: CresciTweetDataset):
    assert isinstance(cresci_genuine_accounts_tweets[19377].text, str)
    assert cresci_genuine_accounts_tweets[19377].text == ""


def test_cresci_user_row_loaded(cresci_genuine_accounts_users: CresciUserDataset):
    assert cresci_genuine_accounts_users[0].statuses_count == 2177
    assert cresci_genuine_accounts_users[0].geo_enabled == True


def test_cresci_user_row_loaded_with_false_values(cresci_genuine_accounts_users: CresciUserDataset):
    assert cresci_genuine_accounts_users[2].statuses_count == 1254
    assert cresci_genuine_accounts_users[2].geo_enabled == False


def test_cresci_all_users_loaded(cresci_genuine_accounts_users: CresciUserDataset):
    assert len(cresci_genuine_accounts_users) == 3474


@modes("cpu", "cuda")
def test_cresci_tensor_tweet_device_is_correct(device: torch.device, cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert cresci_genuine_accounts_tweets_tensors.device == device


@modes("cpu", "cuda")
@pytest.mark.benchmark(group="cresci_tensor_tweet_cached_access", warmup=True)
def test_bench_cresci_tensor_tweet_cached_access(benchmark, cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    result = benchmark(cresci_genuine_accounts_tweets_tensors.__getitem__, 0)
    assert result is not None


@modes("cpu", "cuda")
@pytest.mark.benchmark(group="cresci_tensor_tweet_fresh_access")
def test_bench_cresci_tensor_tweet_fresh_access(benchmark, cresci_genuine_accounts_tweets: CresciTweetDataset, glove_embedding: WordEmbeddings):
    # TODO: Make this benchmark run a different set of indices for each iteration
    dataset = CresciTensorTweetDataset(
        data_source=cresci_genuine_accounts_tweets,
        embeddings=glove_embedding,
        tokenizer=tokenize
    )

    def access():
        return [dataset[i] for i in range(1000)]

    result = benchmark.pedantic(access)

    assert result is not None


@modes("cpu", "cuda")
def test_cresci_tensor_tweet_dataset_is_created(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert cresci_genuine_accounts_tweets_tensors is not None


@modes("cpu", "cuda")
def test_cresci_tensor_tweet_dataset_loads_tensors(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert torch.is_tensor(cresci_genuine_accounts_tweets_tensors[0])


@modes("cpu", "cuda")
def test_cresci_tensor_tweet_encodes_all_words(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert len(cresci_genuine_accounts_tweets_tensors[0]) == 24


@modes("cpu", "cuda")
def test_cresci_tensor_tweet_dataset_returns_word_indices(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert cresci_genuine_accounts_tweets_tensors[0].dtype == torch.long
