from typing import Union, Sequence

import torch
from torch import nn
from torch.nn import Embedding
from torch.nn import functional
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch import Tensor, LongTensor

from sockpuppet.model.embedding import WordEmbeddings


class ContextualLSTM(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, hidden_layers: int=32):
        super(ContextualLSTM, self).__init__()
        self.word_embeddings = word_embeddings

        self.embeddings = self.word_embeddings.to_layer()  # type: Embedding
        self.embeddings.padding_idx = len(self.word_embeddings) - 1

        self.lstm = nn.LSTM(word_embeddings.dim, hidden_layers, batch_first=False)
        self.dense1 = nn.Linear(hidden_layers, 128)
        self.dense2 = nn.Linear(self.dense1.out_features, 64)
        self.output = nn.Linear(self.dense2.out_features, 2)

        # note: there are special data parallelism modules.  read about them!
        # is there a layer that takes the weighted average of two like-shaped tensors? would be useful
        # for mixing the main output and the aux output like the paper describes
        # if not, just mix them myself

    def forward(self, sentences) -> Tensor:
        # IN: List of LongTensors
        # OUT: One FloatTensor

        # TODO: Reset hidden state before each batch
        sentences = sorted(sentences, key=len, reverse=True)
        lengths = LongTensor([len(s) for s in sentences], device=sentences[0].device)

        padded = pad_sequence(sentences, False, self.embeddings.padding_idx)
        # ^ Size([num_tweets, longest_tweet])

        embedding = self.embeddings(padded)
        # ^ Size([num_tweets, longest_tweet, self.word_embeddings.dim])

        packed = pack_padded_sequence(embedding, lengths, False)
        out, (hn, cn) = self.lstm(packed)
        # ^ Size([num_tweets, num_tokens, num_dims]) -> Size([???])
        # TODO: Figure out exactly what the dimensions are
        # out: Output features on every element (word vector) of the input
        # hn: Last element's hidden state
        # cn: Last element's cell state

        a = functional.relu(self.dense1(hn))  # Size([???]) -> Size([???])
        b = functional.relu(self.dense2(a))  # Size([???]) -> Size([???])
        c = torch.sigmoid(self.output(b))  # Size([???]) -> Size([num_tweets, 2])
        return c.squeeze()
        # TODO: May have to add another dimension for each of two classes (yes or no)
        # TODO: What optimizer did the paper use?  What loss function?
