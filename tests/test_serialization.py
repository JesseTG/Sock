import os.path
import lzma
import pytest

from sockpuppet.model.nn import ContextualLSTM
from sockpuppet.model.data import WordEmbeddings
from sockpuppet.model.nn.ContextualLSTM import save, load

from tests.marks import *


def test_save(tmpdir, lstm_cpu: ContextualLSTM):
    path = os.path.join(tmpdir, "test_save.pkl")
    save(lstm_cpu, path)
    assert os.path.exists(path)
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


def test_load_same_device(tmpdir, glove_embedding_cpu: WordEmbeddings, lstm_cpu: ContextualLSTM):
    path = os.path.join(tmpdir, "test_load_same_device.pkl")
    save(lstm_cpu, path)

    assert os.path.exists(path)
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0

    loaded = load(glove_embedding_cpu, path, "cpu")

    assert loaded is not None
    assert isinstance(loaded, ContextualLSTM)
    assert loaded.device.type == "cpu"


@needs_cuda
def test_load_different_device(tmpdir, glove_embedding_cpu: WordEmbeddings, lstm_cuda: ContextualLSTM):
    path = os.path.join(tmpdir, "test_load_different_device.pkl")
    save(lstm_cuda, path)

    assert os.path.exists(path)
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0

    loaded = load(glove_embedding_cpu, path, "cpu")

    assert loaded is not None
    assert isinstance(loaded, ContextualLSTM)
    assert loaded.device.type == "cpu"


def test_saving_doesnt_remove_embeddings(tmpdir, lstm_cpu: ContextualLSTM):
    path = os.path.join(tmpdir, "test_saving_doesnt_remove_embeddings.pkl")
    state = lstm_cpu.state_dict()

    save(lstm_cpu, path)

    assert "embeddings.weight" in state


def test_saved_model_has_same_parameters():
    pytest.xfail()


def test_saved_model_doesnt_have_embeddings():
    pytest.xfail()


def test_metadata_is_preserved():
    pytest.xfail()
