import argparse
import logging
import math
import random
from argparse import ArgumentParser, ArgumentTypeError, FileType

import ignite
import torch
from ignite.engine import Engine, Events, State, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, Timer
from ignite.metrics import BinaryAccuracy, Loss, Precision, Recall
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from sock.model.data import WordEmbeddings, tokenize
from sock.model.data.batching import sentence_label_pad, sentence_pad
from sock.model.dataset import (CresciTweetDataset, Five38TweetDataset, LabelDataset, NbcTweetDataset,
                                SingleLabelDataset, TweetTensorDataset)
from sock.model.nn import ContextualLSTM
from sock.model.serial import load, save
from sock.utils import BOT, NOT_BOT, Metrics, Splits, expand_binary_class, split_integers, to_singleton_row


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Display info about a model"
    )

    parser.add_argument(
        "--glove",
        help="The word vector embeddings to use",
        metavar="path",
        type=FileType('r', encoding="utf8"),
        required=True
    )

    parser.add_argument(
        "model",
        help="A trained model",
        metavar="model",
        type=FileType('rb')
    )

    return parser


def load_glove(args) -> WordEmbeddings:

    embeddings = WordEmbeddings(args.glove, device="cpu")

    return embeddings


def load_model(args, embeddings) -> ContextualLSTM:
    model = load(embeddings, args.model, "cpu")

    return model


def main():
    parser = build_parser()
    args = parser.parse_args()

    glove = load_glove(args)
    model = load_model(args, glove)

    memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model: {args.model.name}")
    print(f"\tTotal Memory: {memory}B")
    for name, param in model.named_parameters():
        print(f"\t{name}:")
        print(f"\t\tSize: {list(param.size())}")
        print(f"\t\tType: {param.dtype}")
        print(f"\t\tElement Size: {param.element_size()}")
        print(f"\t\tElements: {param.numel()}")
        print(f"\t\tMemory: {param.numel() * param.element_size()}B")
        print(f"\t\tRequires Gradient: {param.requires_grad}")
        print(f"\t\tStorage Type: {param.storage_type().__name__}")
        print(f"\t\tStride: {list(param.stride())}")
        print(f"\t\tLayout: {param.layout}")


if __name__ == '__main__':
    main()
