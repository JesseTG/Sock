from typing import Union, Sequence, Tuple, Optional

import torch
from torch.optim import Optimizer

from sockpuppet.model.data import WordEmbeddings
from sockpuppet.model.nn import ContextualLSTM


# All parameters EXCEPT the embeddings are saved to disk
# The embeddings determine the shapes of some parameters, and load_state_dict needs the shapes to be the same

def save(model: ContextualLSTM, out: Union[str]):
    state = model.state_dict()
    del state["embeddings.weight"]
    torch.save(state, out)


def load(embeddings: WordEmbeddings, path, device: torch.device) -> ContextualLSTM:
    model = ContextualLSTM(embeddings, device=device)
    state = torch.load(path, device)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model
