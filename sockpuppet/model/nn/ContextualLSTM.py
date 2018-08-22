from typing import Union, Sequence, Tuple

import torch
from torch import nn
from torch.nn import Embedding
from torch.nn import functional
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch import Tensor, LongTensor

from sockpuppet.model.dataset import PaddedSequence
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

        self.to(device, non_blocking=True)
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

    def forward(self, sentences: Union[Sequence[LongTensor], PaddedSequence]) -> Tensor:
        # IN: List of LongTensors
        # OUT: One FloatTensor

        padded = None
        lengths = None
        if isinstance(sentences, PaddedSequence):
            # If these sentences have already been padded (likely with a DataLoader)...
            padded = sentences.padded
            lengths = sentences.lengths

        elif isinstance(sentences, Sequence) or torch.is_tensor(sentences):
            # Else if this is a plain list of tensors (likely given manually)...
            # We must pack them ourselves
            sorted_sentences = sorted(sentences, key=len, reverse=True)
            lengths = torch.as_tensor([len(s) for s in sorted_sentences], dtype=torch.long, device=self.device)
            padded = pad_sequence(sorted_sentences, True, self.embeddings.padding_idx)
            # ^ Size([num_tweets, longest_tweet])
            # TODO: Replace this part with a call to sentence_pad
        else:
            raise TypeError(f"sentences cannot be a {type(sentences)}")

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

        a = functional.relu(self.dense1(hn))  # Size([???]) -> Size([???])
        b = functional.relu(self.dense2(a))  # Size([???]) -> Size([???])
        c = torch.sigmoid(self.output(b))  # Size([???]) -> Size([num_tweets, 1])
        return c.view(num_sentences)
        # TODO: Consider using BCEWithLogitsLoss
        # TODO: What optimizer did the paper use?  What loss function?
