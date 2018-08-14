import pytest
import torch
from sockpuppet.model.dataset.cresci import CresciTweetDataset, CresciUserDataset, CresciTensorTweetDataset


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


def test_cresci_tensor_tweet_device_is_correct(device, cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert cresci_genuine_accounts_tweets_tensors.device == torch.device(device)


def test_cresci_tensor_tweet_dataset_is_created(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert cresci_genuine_accounts_tweets_tensors is not None


def test_cresci_tensor_tweet_dataset_loads_tensors(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert torch.is_tensor(cresci_genuine_accounts_tweets_tensors[0])


def test_cresci_tensor_tweet_embeds_all_words(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert len(cresci_genuine_accounts_tweets_tensors[0]) == 25


def test_cresci_tensor_tweet_dataset_returns_word_indices(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    assert cresci_genuine_accounts_tweets_tensors[0].dtype == torch.long
