import pytest
import torch
import numpy
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from sockpuppet.model.dataset import sentence_collate
from .marks import *

torch.manual_seed(0)


@pytest.fixture(scope="function")
def dataset(cresci_genuine_accounts_tweets_tensors: Dataset):
    # TODO: Make an alias of this with pytest idioms
    return cresci_genuine_accounts_tweets_tensors


@pytest.fixture(scope="module")
def sampler(cresci_genuine_accounts_tweets: Dataset):
    return SubsetRandomSampler(numpy.arange(len(cresci_genuine_accounts_tweets) // 10000))


@devices("cpu", "cuda")
def test_create_dataloader(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler)
    assert loader is not None


@devices("cpu", "cuda")
def test_create_parallel_dataloader(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, num_workers=4)
    assert loader is not None


@devices("cpu", "cuda")
def test_iterate_dataloader_one_thread(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler)
    tensors = [t for t in loader]

    assert len(tensors) > 0

# TODO: Chokes on SeaWulf


@devices("cpu")
def test_iterate_dataloader_parallel(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, num_workers=2)
    tensors = [t for t in loader]

    assert len(tensors) > 0

# TODO: Test that loading in parallel is faster than loading in sequence
# TODO: Test that pinned memory is faster


@devices("cpu")
def test_dataloader_pin(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, pin_memory=True)
    tensors = [t for t in loader]

    assert len(tensors) > 0
    assert torch.is_tensor(tensors[0])
    assert tensors[0].is_pinned()


@devices("cpu", "cuda")
def test_dataloader_batch(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, collate_fn=sentence_collate, batch_size=8)
    tensors = [t for t in loader]

    assert len(tensors) > 0
    assert torch.is_tensor(tensors[0])
