from typing import Sequence, Tuple, Union

import torch
from torch import LongTensor, Tensor
from torch.nn import LSTM, Embedding, Linear, Module, ReLU, Sequential, Sigmoid, functional
from torch.nn.init import normal_
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence

from sockpuppet.model.data import WordEmbeddings


class ContextualLSTM(Module):
    def __init__(
        self,
        word_embeddings: WordEmbeddings,
        hidden_layers: int=32,
        device: Union[torch.device, str]="cpu"
    ):
        super(ContextualLSTM, self).__init__()
        self.word_embeddings = word_embeddings

        self.embeddings = self.word_embeddings.to_layer()  # type: Embedding
        self.embeddings.padding_idx = 0

        self.lstm = LSTM(word_embeddings.dim, hidden_layers, batch_first=False)

        self.output = Sequential(
            Linear(hidden_layers, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 1),
            Sigmoid()
        )

        self.to(device, non_blocking=True)
        # is there a layer that takes the weighted average of two like-shaped tensors? would be useful
        # for mixing the main output and the aux output like the paper describes
        # if not, just mix them myself

    @property
    def device(self) -> torch.device:
        return self.embeddings.weight.device

    def _init_hidden(self, batch_size) -> Tuple[Tensor, Tensor]:
        def make_zeros():
            return torch.zeros(
                self.lstm.num_layers,
                batch_size,
                self.lstm.hidden_size,
                dtype=torch.float,
                device=self.device
            )

        return (make_zeros(), make_zeros())

    def extra_repr(self) -> str:
        return f"<device>: {self.device}"

    def forward(self, sentences: Sequence[LongTensor]) -> Tensor:
        padded = sentences[0]
        lengths = sentences[1]
        num_sentences = len(lengths)
        self.hidden = self._init_hidden(num_sentences)
        embedding = self.embeddings(padded)
        # ^ Size([num_tweets, longest_tweet, self.word_embeddings.dim])

        packed = pack_padded_sequence(embedding, lengths, True)
        self.lstm.flatten_parameters()
        # NOTE: Don't know what this does, need to ask around

        out, (hn, cn) = self.lstm(packed, self.hidden)
        # ^ Size([num_tweets, num_tokens, num_dims]) -> Size([???])
        # TODO: Figure out exactly what the dimensions are
        # out: Output features on every element (word vector) of the input
        # hn: Last element's hidden state
        # cn: Last element's cell state

        hn = hn.view(num_sentences, self.lstm.hidden_size)
        # Only using one LSTM layer

        result = self.output(hn)

        return result.view(num_sentences)
        # a = functional.relu(self.dense1(hn))  # Size([???]) -> Size([???])
        # b = functional.relu(self.dense2(a))  # Size([???]) -> Size([???])
        # c = torch.sigmoid(self.output(b))  # Size([???]) -> Size([num_tweets, 1])
        # return c.view(num_sentences)
        # TODO: Consider using BCEWithLogitsLoss
        # TODO: What optimizer did the paper use?  What loss function?


def save(model: ContextualLSTM, out: Union[str]):
    state = model.state_dict()  # type: dict
    del state["embeddings.weight"]
    torch.save(state, out)


def load(embeddings: WordEmbeddings, path, device) -> ContextualLSTM:
    model = ContextualLSTM(embeddings)
    state = torch.load(path, device)
    model.load_state_dict(state, strict=False)
    model.train(False)

    return model
