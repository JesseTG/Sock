from typing import Sequence, Union, Tuple
import torch
from torch import Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence


def sentence_collate(sentences: Sequence[LongTensor]) -> LongTensor:
    sentences = sorted(sentences, key=len, reverse=True)
    padded = pad_sequence(sentences, False, 0)

    return padded


def sentence_collate_batch(sentences: Sequence[Tuple[LongTensor, int]]) -> Tuple[LongTensor, Sequence[int]]:
    sentences = sorted(sentences, key=lambda s: len(s[0]), reverse=True)
    encodings = [s[0] for s in sentences]
    labels = torch.as_tensor([s[1] for s in sentences], dtype=torch.float, device=sentences[0][0].device)

    padded = pad_sequence(encodings, True, 0)

    return padded, labels
