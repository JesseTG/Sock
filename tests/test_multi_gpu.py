from typing import Callable, Sequence, Tuple
from collections import namedtuple
import time

import pytest
import torch
import ignite

from torch import Tensor, LongTensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import DataParallel
from ignite.engine import Events, Engine
import ignite.metrics
from sockpuppet.model.nn.ContextualLSTM import ContextualLSTM
from sockpuppet.model.embedding import WordEmbeddings
from sockpuppet.model.dataset import LabelDataset, sentence_collate_batch
from tests.marks import needs_cuda, needs_cudnn, needs_multiple_gpus, cuda_only


torch.manual_seed(0)


@needs_cuda
@cuda_only
@needs_multiple_gpus
def test_create_data_parallel_lstm(device, lstm: ContextualLSTM):
    # TODO: This test is running on my laptop, but it shouldn't as I only have one GPU
    dataparallel = DataParallel(lstm)

    assert dataparallel is not None


@needs_cuda
@cuda_only
@needs_multiple_gpus
def test_data_parallel_lstm_evaluates(lstm: ContextualLSTM):
    encoding = [
        torch.tensor([0, 1, 5, 8, 3, 1], dtype=torch.long, device=lstm.device),
        torch.tensor([1, 4, 6, 1, 9, 7], dtype=torch.long, device=lstm.device),
        torch.tensor([9, 0, 6, 9, 9, 0], dtype=torch.long, device=lstm.device),
        torch.tensor([2, 3, 6, 1, 2, 4], dtype=torch.long, device=lstm.device),
    ]

    dataparallel = DataParallel(lstm)
    result = dataparallel(encoding)

    assert result is not None


@needs_multiple_gpus
@needs_cuda
@cuda_only
def test_data_parallel_returns_same_result_as_serial(lstm: ContextualLSTM, glove_embedding: WordEmbeddings):
    encoding = [
        torch.tensor([0, 1, 5, 8, 3, 1], dtype=torch.long, device=lstm.device),
        torch.tensor([1, 4, 6, 1, 9, 7], dtype=torch.long, device=lstm.device),
        torch.tensor([9, 0, 6, 9, 9, 0], dtype=torch.long, device=lstm.device),
        torch.tensor([2, 3, 6, 1, 2, 4], dtype=torch.long, device=lstm.device),
    ]
    serial = ContextualLSTM(glove_embedding, device="cuda")
    serial_result = serial(encoding)

    parallel = DataParallel(lstm)
    parallel_result = parallel(encoding)

    assert serial_result.cpu().numpy() == pytest.approx(parallel_result.cpu().numpy())


@pytest.mark.skip
@needs_cuda
@needs_multiple_gpus
def test_train_data_parallel_lstm():
    assert 0


@pytest.mark.skip
@needs_cuda
@needs_multiple_gpus
def test_train_distributed_lstm():
    assert 0

# TODO: Structure these properly; make one test to compare times depend on all of these
# see pytest-dependency


@pytest.mark.skip
@needs_cuda
@needs_multiple_gpus
def test_multi_cuda_faster_than_1_cuda():
    assert 0


@pytest.mark.skip
def test_distributed_faster_than_multi_cuda():
    assert 0
