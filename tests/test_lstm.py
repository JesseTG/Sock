import pytest
import torch
from sockpuppet.model.nn import ContextualLSTM
from sockpuppet.model.embedding import WordEmbeddings
from tests.marks import needs_cuda, needs_cudnn, cuda_only


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


@cuda_only
@needs_cuda
def test_lstm_moves_all_data_to_cuda(lstm: ContextualLSTM):
    for p in lstm.parameters():
        assert p.is_cuda


@cuda_only
@needs_cuda
def test_lstm_moves_embeddings_to_cuda(lstm: ContextualLSTM):
    assert lstm.embeddings.weight.is_cuda


@cuda_only
@needs_cuda
def test_lstm_needs_input_from_same_device(lstm: ContextualLSTM):
    with pytest.raises(RuntimeError):
        encoding = [
            torch.LongTensor([0, 1, 5, 78, 3, 1])
        ]
        assert encoding[0].device.type == "cpu"

        lstm(encoding)


def test_lstm_evaluates(lstm: ContextualLSTM):
    encoding = [
        torch.tensor([7, 1, 5, 78, 3, 1], dtype=torch.long, device=lstm.device)
    ]
    assert encoding[0].device == lstm.device

    result = lstm(encoding)
    assert torch.is_tensor(result)
    assert result.device == lstm.device


def test_lstm_rejects_list_of_lists(lstm: ContextualLSTM):
    encoding = [
        [0, 1, 5, 8, 3, 1],
        [1, 4, 6, 1, 9, 7],
        [9, 0, 6, 9, 9, 0],
        [2, 3, 6, 1, 2, 4],
    ]

    with pytest.raises(Exception):
        result = lstm(encoding)


def test_lstm_evaluates_batches_of_same_length_in_tensor(lstm: ContextualLSTM):
    encoding = torch.tensor([
        [0, 1, 5, 8, 3, 1],
        [1, 4, 6, 1, 9, 7],
        [9, 0, 6, 9, 9, 0],
        [2, 3, 6, 1, 2, 4],
    ], dtype=torch.long, device=lstm.device)

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_evaluates_batches_of_same_length_in_list_of_tensors(lstm: ContextualLSTM):
    encoding = [
        torch.tensor([0, 1, 5, 8, 3, 1], dtype=torch.long, device=lstm.device),
        torch.tensor([1, 4, 6, 1, 9, 7], dtype=torch.long, device=lstm.device),
        torch.tensor([9, 0, 6, 9, 9, 0], dtype=torch.long, device=lstm.device),
        torch.tensor([2, 3, 6, 1, 2, 4], dtype=torch.long, device=lstm.device),
    ]

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_evaluates_batches_of_different_length_in_list_of_tensors_unsorted(lstm: ContextualLSTM):
    encoding = [
        torch.tensor([0, 1, 5, 8, 3], dtype=torch.long, device=lstm.device),
        torch.tensor([1, 4, 6, 1, 9, 7, 9, 1], dtype=torch.long, device=lstm.device),
        torch.tensor([9, 0, 6, 9], dtype=torch.long, device=lstm.device),
        torch.tensor([2, 3, 6, 1, 2, 4, 4], dtype=torch.long, device=lstm.device),
    ]

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_evaluates_batches_of_different_length_in_list_of_tensors_sorted(lstm: ContextualLSTM):
    encoding = [
        torch.tensor([1, 4, 6, 1, 9, 7, 9, 1], dtype=torch.long, device=lstm.device),
        torch.tensor([2, 3, 6, 1, 2, 4, 4], dtype=torch.long, device=lstm.device),
        torch.tensor([0, 1, 5, 8, 3], dtype=torch.long, device=lstm.device),
        torch.tensor([9, 0, 6, 9], dtype=torch.long, device=lstm.device),
    ]

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_returns_1d_float_tensor_from_list_of_tensors(lstm: ContextualLSTM):
    encoding = [
        torch.tensor([0, 1, 5, 8, 3, 1], dtype=torch.long, device=lstm.device),
        torch.tensor([1, 4, 6, 1, 9, 7], dtype=torch.long, device=lstm.device),
        torch.tensor([9, 0, 6, 9, 9, 0], dtype=torch.long, device=lstm.device),
        torch.tensor([2, 3, 6, 1, 2, 4], dtype=torch.long, device=lstm.device),
    ]

    result = lstm(encoding)
    assert result.dtype.is_floating_point
    assert result.shape == torch.Size([len(encoding)])


def test_lstm_returns_1d_float_tensor_from_tensor(lstm: ContextualLSTM, something):
    encoding = torch.tensor([
        [0, 1, 5, 8, 3, 1],
        [1, 4, 6, 1, 9, 7],
        [9, 0, 6, 9, 9, 0],
        [2, 3, 6, 1, 2, 4],
    ], dtype=torch.long, device=lstm.device)

    result = lstm(encoding)
    assert result.dtype.is_floating_point
    assert result.shape == torch.Size([len(encoding)])
