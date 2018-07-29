import pytest
import torch
from sockpuppet.model.nn.ContextualLSTM import ContextualLSTM
from sockpuppet.model.embedding import WordEmbeddings
from tests.marks import needs_cuda, needs_cudnn


@pytest.fixture(scope="function")
def lstm(glove_embedding: WordEmbeddings):
    lstm = ContextualLSTM(glove_embedding)
    lstm = lstm.cpu()
    return lstm


@pytest.fixture(scope="function")
def lstm_cuda(glove_embedding: WordEmbeddings):
    lstm = ContextualLSTM(glove_embedding)
    lstm = lstm.cuda()
    torch.cuda.synchronize()
    return lstm


def test_create_lstm(lstm: ContextualLSTM):
    assert lstm is not None


def test_has_modules(lstm: ContextualLSTM):
    modules = tuple(lstm.modules())
    assert modules != []


def test_has_parameters(lstm: ContextualLSTM):
    parameters = tuple(lstm.parameters())
    assert parameters != []


@needs_cuda
def test_lstm_moves_all_data_to_cuda(lstm_cuda: ContextualLSTM):
    for p in lstm_cuda.parameters():
        assert p.is_cuda


@needs_cuda
def test_lstm_moves_embeddings_to_cuda(lstm_cuda: ContextualLSTM):
    assert lstm_cuda.embeddings.weight.is_cuda


@needs_cuda
def test_lstm_needs_input_from_same_device(lstm_cuda: ContextualLSTM):
    with pytest.raises(RuntimeError):
        encoding = [
            torch.LongTensor([0, 1, 5, 78, 3, 1])
        ]
        assert encoding[0].device.type == "cpu"

        lstm_cuda(encoding)


@needs_cuda
@needs_cudnn
def test_lstm_evaluates_on_cuda(lstm_cuda: ContextualLSTM):
    encoding = [
        torch.tensor([0, 1, 5, 78, 3, 1], dtype=torch.long, device=torch.device("cuda"))
    ]
    torch.cuda.synchronize()
    assert encoding[0].is_cuda

    result = lstm_cuda(encoding)
    torch.cuda.synchronize()
    assert torch.is_tensor(result)
    assert result.is_cuda


def test_lstm_evaluates_on_cpu(lstm: ContextualLSTM):
    encoding = [
        torch.LongTensor([7, 1, 5, 78, 3, 1])
    ]
    assert encoding[0].device.type == "cpu"

    result = lstm(encoding)
    assert torch.is_tensor(result)
    assert result.device.type == "cpu"


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
    encoding = torch.LongTensor([
        [0, 1, 5, 8, 3, 1],
        [1, 4, 6, 1, 9, 7],
        [9, 0, 6, 9, 9, 0],
        [2, 3, 6, 1, 2, 4],
    ])

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_evaluates_batches_of_same_length_in_list_of_tensors(lstm: ContextualLSTM):
    encoding = [
        torch.LongTensor([0, 1, 5, 8, 3, 1]),
        torch.LongTensor([1, 4, 6, 1, 9, 7]),
        torch.LongTensor([9, 0, 6, 9, 9, 0]),
        torch.LongTensor([2, 3, 6, 1, 2, 4]),
    ]

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_evaluates_batches_of_different_length_in_list_of_tensors_unsorted(lstm: ContextualLSTM):
    encoding = [
        torch.LongTensor([0, 1, 5, 8, 3]),
        torch.LongTensor([1, 4, 6, 1, 9, 7, 9, 1]),
        torch.LongTensor([9, 0, 6, 9]),
        torch.LongTensor([2, 3, 6, 1, 2, 4, 4]),
    ]

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_evaluates_batches_of_different_length_in_list_of_tensors_sorted(lstm: ContextualLSTM):
    encoding = [
        torch.LongTensor([1, 4, 6, 1, 9, 7, 9, 1]),
        torch.LongTensor([2, 3, 6, 1, 2, 4, 4]),
        torch.LongTensor([0, 1, 5, 8, 3]),
        torch.LongTensor([9, 0, 6, 9]),
    ]

    result = lstm(encoding)
    assert torch.is_tensor(result)


def test_lstm_returns_2d_tensor(lstm: ContextualLSTM):
    encoding = [
        torch.LongTensor([0, 1, 5, 8, 3, 1]),
        torch.LongTensor([1, 4, 6, 1, 9, 7]),
        torch.LongTensor([9, 0, 6, 9, 9, 0]),
        torch.LongTensor([2, 3, 6, 1, 2, 4]),
    ]

    result = lstm(encoding)
    assert result.dtype.is_floating_point
    assert result.shape == torch.Size([len(encoding), 2])
