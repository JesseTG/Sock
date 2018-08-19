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
    return SubsetRandomSampler(range(512))


@pytest.mark.parametrize("num_workers", [0, 1, 4])
@modes("cpu", "cuda")
def test_create_dataloader(dataset: Dataset, sampler: Sampler, num_workers: int):
    loader = DataLoader(dataset=dataset, sampler=sampler, num_workers=num_workers)
    assert loader is not None


@modes("cpu", "cuda")
@pytest.mark.parametrize("num_workers", [0, 1])
def test_iterate_dataloader_one_thread(dataset: Dataset, sampler: Sampler, num_workers: int):
    loader = DataLoader(dataset=dataset, sampler=sampler, num_workers=num_workers)
    tensors = [t for t in loader]

    assert len(tensors) > 0


@modes("cpu")
@pytest.mark.parametrize("pin_memory", [False, True], ids=["unpinned", "pinned"])
@pytest.mark.parametrize("num_workers", [0, 1, 2, 4], ids=lambda x: f"{x}t")
@pytest.mark.parametrize("batch_size", [1, 8, 64, 256], ids=lambda x: f"{x}b")
@pytest.mark.benchmark(group="dataloader_iteration", warmup=True)
def test_bench_dataloader_iteration(benchmark, dataset: Dataset, sampler: Sampler, pin_memory: bool, num_workers: int, batch_size: int):
    def iterate():
        loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size=batch_size
        )

        tensors = [t for t in loader]

    result = benchmark(iterate)


@modes("cpu")
def test_dataloader_pin(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, pin_memory=True)
    tensors = [t for t in loader]

    assert len(tensors) > 0
    assert torch.is_tensor(tensors[0])
    assert tensors[0].is_pinned()


@modes("cpu", "cuda")
def test_dataloader_batch(dataset: Dataset, sampler: Sampler):
    loader = DataLoader(dataset=dataset, sampler=sampler, collate_fn=sentence_collate, batch_size=8)
    tensors = [t for t in loader]

    assert len(tensors) > 0
    assert torch.is_tensor(tensors[0])
