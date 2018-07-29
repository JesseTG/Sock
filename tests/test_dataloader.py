import pytest
import torch
import numpy
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler


@pytest.fixture(scope="module")
def dataset(cresci_genuine_accounts_tweets_tensors: Dataset):
    return cresci_genuine_accounts_tweets_tensors


@pytest.fixture(scope="module")
def sampler(dataset: Dataset):
    return SubsetRandomSampler(numpy.arange(len(dataset) // 10000))


def test_create_dataloader(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler)
    assert loader is not None


def test_create_parallel_dataloader(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, num_workers=4)
    assert loader is not None


def test_iterate_dataloader_one_thread(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler)
    tensors = [t for t in loader]

    assert len(tensors) > 0


def test_iterate_dataloader_parallel(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, num_workers=4)
    tensors = [t for t in loader]

    assert len(tensors) > 0


def test_dataloader_pin(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, pin_memory=True)
    tensors = [t for t in loader]

    assert len(tensors) > 0
    assert torch.is_tensor(tensors[0])


def test_dataloader_batch(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=10)
    tensors = [t for t in loader]

    assert len(tensors) > 0
    assert torch.is_tensor(tensors[0])
