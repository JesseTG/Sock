from typing import Sequence
from torch import Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence


def sentence_collate(sentences: Sequence[LongTensor]) -> LongTensor:
    sentences = sorted(sentences, key=len, reverse=True)
    padded = pad_sequence(sentences, False, 0)

    return padded
