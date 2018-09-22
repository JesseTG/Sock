import os.path
import lzma
import pytest
import shutil

import torch

from sockpuppet.model.nn import ContextualLSTM
from sockpuppet.model.data import WordEmbeddings
from sockpuppet.model.serial import save, load

from tests.marks import *


@pytest.fixture
def path(request, tmpdir):
    p = os.path.join(tmpdir, f"{request.node.name}.pt")
    yield p
    if "keep_saved" not in request.keywords:
        shutil.rmtree(tmpdir)


@modes("cpu", "cuda")
@keep_saved
def test_save(path, lstm: ContextualLSTM):
    save(lstm, path)
    assert os.path.exists(path)
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


@modes("cpu", "cuda")
def test_load_same_device(path, glove_embedding: WordEmbeddings, lstm: ContextualLSTM, device: torch.device):
    save(lstm, path)

    loaded = load(glove_embedding, path, device)

    assert loaded is not None
    assert isinstance(loaded, ContextualLSTM)
    assert loaded.device.type == device.type


@needs_cuda
def test_load_different_device(path, glove_embedding_cpu: WordEmbeddings, lstm_cuda: ContextualLSTM):
    save(lstm_cuda, path)

    loaded = load(glove_embedding_cpu, path, "cpu")

    assert loaded is not None
    assert isinstance(loaded, ContextualLSTM)
    assert loaded.device.type == "cpu"


@modes("cpu", "cuda")
def test_saving_doesnt_remove_embeddings(path, lstm: ContextualLSTM):
    state = lstm.state_dict()

    save(lstm, path)

    assert "embeddings.weight" in state


@modes("cpu", "cuda")
def test_saved_model_has_same_parameters(path, glove_embedding: WordEmbeddings, lstm: ContextualLSTM, device: torch.device):
    save(lstm, path)

    loaded = load(glove_embedding, path, device)

    loaded_params = [p for p in loaded.output.parameters()]
    saved_params = [p for p in lstm.output.parameters()]
    # Don't compare the embeddings; too expensive

    for old, new in zip(loaded_params, saved_params):
        assert old.detach().cpu().numpy() == pytest.approx(new.detach().cpu().numpy())


@modes("cpu", "cuda")
def test_saved_model_doesnt_have_embeddings(path, lstm: ContextualLSTM, device: torch.device):
    save(lstm, path)

    state = torch.load(path, device)

    assert "embeddings.weight" not in state
