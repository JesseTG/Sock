import pytest
import torch
from pandas import DataFrame
from torch.nn import Embedding
from sockpuppet.model.embedding import WordEmbeddings


from tests.marks import *

FIRST_ROW_VECTOR = torch.as_tensor([
    0.62415, 0.62476, -0.082335, 0.20101, -0.13741, -0.11431, 0.77909, 2.6356, -0.46351, 0.57465,
    -0.024888, -0.015466, -2.9696, -0.49876, 0.095034, -0.94879, -0.017336, -0.86349, -1.3348, 0.046811,
    0.36999, -0.57663, -0.48469, 0.40078, 0.75345
], dtype=torch.float)

ZERO_VECTOR = torch.zeros_like(FIRST_ROW_VECTOR)


@pytest.fixture(scope="module")
def embedding_layer(glove_embedding: WordEmbeddings):
    return Embedding.from_pretrained(glove_embedding.vectors)


def test_correct_embedding_words_loaded(glove_data: DataFrame):
    assert glove_data[0][2] == "<user>"


@modes("cpu", "cuda")
def test_all_embedding_vectors_loaded(glove_embedding: WordEmbeddings):
    assert len(glove_embedding) == 1193516


def test_pad_is_index_0(glove_data: DataFrame):
    assert glove_data[0][0] == "<pad>"


def test_unk_is_index_1(glove_data: DataFrame):
    assert glove_data[0][1] == "<unk>"


@modes("cpu", "cuda")
def test_first_word_vector_is_all_zeros(glove_embedding: WordEmbeddings):
    assert glove_embedding[0].cpu().numpy() == pytest.approx(ZERO_VECTOR.numpy())


@modes("cpu", "cuda")
def test_correct_embedding_vector_length(glove_embedding: WordEmbeddings):
    assert len(glove_embedding.vectors[0]) == 25


@modes("cpu", "cuda")
def test_correct_embedding_values_loaded(glove_embedding: WordEmbeddings):
    assert glove_embedding.vectors[2].cpu().numpy() == pytest.approx(FIRST_ROW_VECTOR.numpy())


@modes("cpu", "cuda")
def test_embedding_length_consistent(glove_embedding: WordEmbeddings, glove_data: DataFrame):
    assert len(glove_embedding.vectors) == len(glove_data)


@modes("cpu", "cuda")
def test_get_vector_by_int_index(glove_embedding: WordEmbeddings):
    assert glove_embedding[2].cpu().numpy() == pytest.approx(FIRST_ROW_VECTOR.numpy())


@modes("cpu", "cuda")
def test_get_vector_by_str_index(glove_embedding: WordEmbeddings):
    assert glove_embedding["<user>"].cpu().numpy() == pytest.approx(FIRST_ROW_VECTOR.numpy())


@modes("cpu", "cuda")
def test_encode_returns_tensor(glove_embedding: WordEmbeddings):
    tokens = "<user> it is not in my video".split()
    encoding = glove_embedding.encode(tokens)

    assert torch.is_tensor(encoding)


@modes("cpu", "cuda")
def test_encode_has_correct_value(glove_embedding: WordEmbeddings):
    tokens = "<user> it is not in my video".split()
    encoding = glove_embedding.encode(tokens)

    assert torch.equal(encoding, torch.tensor([2, 35, 34, 80, 37, 31, 288],
                                              dtype=torch.long, device=glove_embedding.device))


@modes("cpu", "cuda")
def test_unknown_word_embeds_to_zero_vector(glove_embedding: WordEmbeddings):
    embedding = glove_embedding["<france>"]

    assert embedding.cpu().numpy() == pytest.approx(ZERO_VECTOR.numpy())


@modes("cpu", "cuda")
def test_unknown_word_encodes_to_index_1(glove_embedding: WordEmbeddings):
    tokens = "<france> <spain> <china> <user>".split()
    encoding = glove_embedding.encode(tokens)

    assert torch.equal(encoding, torch.as_tensor([1, 1, 1, 2], dtype=torch.long, device=glove_embedding.device))


@modes("cpu", "cuda")
def test_bench_encode(benchmark, glove_embedding: WordEmbeddings):
    tokens = "<user> it is not in my video".split()
    result = benchmark(glove_embedding.encode, tokens)

    assert result is not None


@modes("cpu", "cuda")
def test_encode_empty_string_to_zero(glove_embedding: WordEmbeddings):
    tokens = "".split()
    embedding = glove_embedding.encode(tokens)

    assert embedding[0].cpu().numpy() == pytest.approx(ZERO_VECTOR.numpy())


@modes("cpu", "cuda")
def test_embedding_can_create_layer(glove_embedding: WordEmbeddings):
    layer = glove_embedding.to_layer()
    assert isinstance(layer, Embedding)


@modes("cpu", "cuda")
def test_embedding_layer_can_embed_words(glove_embedding: WordEmbeddings):
    tokens = "<user> it is not in my video".split()
    encoding = glove_embedding.encode(tokens)
    layer = glove_embedding.to_layer()

    assert layer(encoding).cpu().numpy()[0] == pytest.approx(FIRST_ROW_VECTOR.numpy())
