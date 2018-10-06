from collections import namedtuple
from typing import Sequence, Tuple, Union

import torch
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pack_sequence, pad_sequence

PackedSentences = namedtuple("PackedSentences", ["packed", "lengths"])
PaddedSequence = namedtuple("PaddedSequence", ["data", "lengths"])
# TODO: Renamed to PaddedBatch


def sentence_collate(sentences: Sequence[LongTensor]) -> LongTensor:
    sentences = sorted(sentences, key=len, reverse=True)
    padded = pad_sequence(sentences, True, 0)

    return padded


def sentence_pad(sentences: Sequence[LongTensor]) -> PaddedSequence:
    sorted_sentences = sorted(sentences, key=len, reverse=True)
    padded = pad_sequence(sorted_sentences, True, 0)
    lengths = torch.as_tensor([len(s) for s in sorted_sentences], dtype=torch.long, device=padded.device)

    return PaddedSequence(padded, lengths)


def sentence_label_collate(sentences: Sequence[Tuple[LongTensor, int]]) -> Tuple[LongTensor, Tensor]:
    sentences = sorted(sentences, key=lambda s: len(s[0]), reverse=True)
    encodings = [s[0] for s in sentences]
    labels = torch.as_tensor([s[1] for s in sentences], dtype=torch.float, device=sentences[0][0].device)

    padded = pad_sequence(encodings, True, 0)

    return padded, labels


def sentence_label_pad(sentences: Sequence[Tuple[LongTensor, int]]) -> Tuple[PaddedSequence, Tensor]:
    sorted_sentences = sorted(sentences, key=lambda s: len(s[0]), reverse=True)
    encodings = [s[0] for s in sorted_sentences]
    padded = pad_sequence(encodings, True, 0)
    labels = torch.as_tensor([s[1] for s in sorted_sentences], dtype=torch.float, device=padded.device)
    lengths = torch.as_tensor([len(s[0]) for s in sorted_sentences], dtype=torch.long, device=padded.device)

    return PaddedSequence(padded, lengths), labels
