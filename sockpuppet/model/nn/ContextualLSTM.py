from typing import Union, Sequence, Tuple

import torch
from torch import nn
from torch.nn import Embedding
from torch.nn import functional
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch import Tensor, LongTensor

from sockpuppet.model.embedding import WordEmbeddings


class ContextualLSTM(nn.Module):
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

        self.lstm = nn.LSTM(word_embeddings.dim, hidden_layers, batch_first=False)
        self.dense1 = nn.Linear(hidden_layers, 128)
        self.dense2 = nn.Linear(self.dense1.out_features, 64)
        self.output = nn.Linear(self.dense2.out_features, 1)

        self.to(device)
        # is there a layer that takes the weighted average of two like-shaped tensors? would be useful
        # for mixing the main output and the aux output like the paper describes
        # if not, just mix them myself

    @property
    def device(self) -> torch.device:
        return self.dense1.weight.device

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

    def forward(self, sentences) -> Tensor:
        # IN: List of LongTensors
        # OUT: One FloatTensor
        self.hidden = self._init_hidden(len(sentences))

        sentences = sorted(sentences, key=len, reverse=True)
        lengths = torch.as_tensor([len(s) for s in sentences], dtype=torch.long, device=self.device)

        padded = pad_sequence(sentences, False, self.embeddings.padding_idx)
        # ^ Size([num_tweets, longest_tweet])

        embedding = self.embeddings(padded)
        # ^ Size([num_tweets, longest_tweet, self.word_embeddings.dim])

        packed = pack_padded_sequence(embedding, lengths, False)
        out, (hn, cn) = self.lstm(packed, self.hidden)
        # ^ Size([num_tweets, num_tokens, num_dims]) -> Size([???])
        # TODO: Figure out exactly what the dimensions are
        # out: Output features on every element (word vector) of the input
        # hn: Last element's hidden state
        # cn: Last element's cell state

        hn = hn.view(len(sentences), self.lstm.hidden_size)
        # Only using one LSTM layer

        a = functional.relu(self.dense1(hn))  # Size([???]) -> Size([???])
        b = functional.relu(self.dense2(a))  # Size([???]) -> Size([???])
        c = torch.sigmoid(self.output(b))  # Size([???]) -> Size([num_tweets, 1])
        return c.view(len(sentences))
        # TODO: Consider using BCEWithLogitsLoss
        # TODO: What optimizer did the paper use?  What loss function?
