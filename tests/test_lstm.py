import pytest
import torch
from sockpuppet.model.nn import ContextualLSTM
from sockpuppet.model.data import sentence_label_pad, sentence_pad, WordEmbeddings

from tests.marks import *


@modes("cpu", "cuda")
def test_devices_are_the_same(lstm: ContextualLSTM, glove_embedding: WordEmbeddings):
    assert lstm.device == glove_embedding.device


def test_create_lstm(lstm: ContextualLSTM):
    assert lstm is not None


def test_has_modules(lstm: ContextualLSTM):
    modules = tuple(lstm.modules())
    assert modules != []


def test_has_parameters(lstm: ContextualLSTM):
    parameters = tuple(lstm.parameters())
    assert parameters != []


@modes("cuda", "dp")
def test_lstm_moves_all_data_to_cuda(lstm: ContextualLSTM):
    for p in lstm.parameters():
        assert p.is_cuda


@modes("cuda")
def test_lstm_moves_embeddings_to_cuda(lstm_cuda: ContextualLSTM):
    assert lstm_cuda.embeddings.weight.is_cuda


@modes("dp")
def test_lstm_moves_embeddings_to_cuda_in_dp_mode(lstm_dp):
    assert lstm_dp.module.embeddings.weight.is_cuda


@modes("cuda", "dp")
def test_lstm_needs_input_from_same_device(lstm: ContextualLSTM):
    with pytest.raises(RuntimeError):
        encoding = sentence_pad([
            torch.tensor([0, 1, 5, 78, 3, 1], dtype=torch.long, device="cpu")
        ])

        lstm(encoding)


def test_lstm_evaluates(lstm: ContextualLSTM, device: torch.device):
    encoding = sentence_pad([
        torch.tensor([7, 1, 5, 78, 3, 1], dtype=torch.long, device=device)
    ])

    result = lstm(encoding)
    assert torch.is_tensor(result)
    assert result.device == device


@pytest.mark.benchmark(group="test_bench_lstm_evaluates")
def test_bench_lstm_evaluates(benchmark, lstm: ContextualLSTM, device: torch.device):
    encoding = sentence_pad([
        torch.tensor([7, 1, 5, 78, 3, 1], dtype=torch.long, device=device)
    ] * 1000)

    result = benchmark(lstm, encoding)
    assert torch.is_tensor(result)
    assert result.device == device


def test_lstm_rejects_list_of_lists(lstm: ContextualLSTM):
    encoding = [
        [0, 1, 5, 8, 3, 1],
        [1, 4, 6, 1, 9, 7],
        [9, 0, 6, 9, 9, 0],
        [2, 3, 6, 1, 2, 4],
    ]

    with pytest.raises(Exception):
        result = lstm(encoding)


def test_lstm_rejects_tensor(lstm: ContextualLSTM, device: torch.device):
    encoding = torch.tensor([
        [0, 1, 5, 8, 3, 1],
        [1, 4, 6, 1, 9, 7],
        [9, 0, 6, 9, 9, 0],
        [2, 3, 6, 1, 2, 4],
    ], dtype=torch.long, device=device)

    with pytest.raises(Exception):
        result = lstm(encoding)


def test_lstm_evaluates_batches_of_same_length(lstm: ContextualLSTM, device: torch.device):
    encoding = sentence_pad([
        torch.tensor([0, 1, 5, 8, 3, 1], dtype=torch.long, device=device),
        torch.tensor([1, 4, 6, 1, 9, 7], dtype=torch.long, device=device),
        torch.tensor([9, 0, 6, 9, 9, 0], dtype=torch.long, device=device),
        torch.tensor([2, 3, 6, 1, 2, 4], dtype=torch.long, device=device),
    ])

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_evaluates_batches_of_different_length_unsorted(lstm: ContextualLSTM, device: torch.device):
    encoding = sentence_pad([
        torch.tensor([0, 1, 5, 8, 3], dtype=torch.long, device=device),
        torch.tensor([1, 4, 6, 1, 9, 7, 9, 1], dtype=torch.long, device=device),
        torch.tensor([9, 0, 6, 9], dtype=torch.long, device=device),
        torch.tensor([2, 3, 6, 1, 2, 4, 4], dtype=torch.long, device=device),
    ])

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_evaluates_batches_of_different_length_in_sorted(lstm: ContextualLSTM, device: torch.device):
    encoding = sentence_pad([
        torch.tensor([1, 4, 6, 1, 9, 7, 9, 1], dtype=torch.long, device=device),
        torch.tensor([2, 3, 6, 1, 2, 4, 4], dtype=torch.long, device=device),
        torch.tensor([0, 1, 5, 8, 3], dtype=torch.long, device=device),
        torch.tensor([9, 0, 6, 9], dtype=torch.long, device=device),
    ])

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_returns_1d_float_tensor(lstm: ContextualLSTM, device: torch.device):
    encoding = sentence_pad([
        torch.tensor([0, 1, 5, 8, 3, 1], dtype=torch.long, device=device),
        torch.tensor([1, 4, 6, 1, 9, 7], dtype=torch.long, device=device),
        torch.tensor([9, 0, 6, 9, 9, 0], dtype=torch.long, device=device),
        torch.tensor([2, 3, 6, 1, 2, 4], dtype=torch.long, device=device),
    ])

    result = lstm(encoding)
    assert result.dtype.is_floating_point
    assert result.shape == torch.Size([len(encoding[0])])


def test_get_lstm_cpu(request, lstm_cpu: ContextualLSTM):
    assert lstm_cpu is not None
    assert type(lstm_cpu) == ContextualLSTM
    assert lstm_cpu.device.type == "cpu"
