import random
import time
from collections import namedtuple
from itertools import product
from typing import Callable, Dict, Sequence

import ignite
import pytest
import torch
from ignite.engine import Engine, Events, State
from ignite.handlers import EarlyStopping, Timer
from ignite.metrics import BinaryAccuracy, Loss, Precision, Recall
from tests.marks import *
from torch import LongTensor, Tensor
from torch.nn import DataParallel, Module
from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Optimizer, RMSprop, Rprop
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, Subset, TensorDataset, random_split

from sockpuppet.model.data import WordEmbeddings, sentence_label_pad, sentence_pad
from sockpuppet.model.dataset import CresciTensorTweetDataset, Five38TweetTensorDataset, NbcTweetTensorDataset
from sockpuppet.model.dataset.label import LabelDataset, SingleLabelDataset
from sockpuppet.model.nn import ContextualLSTM
from sockpuppet.utils import Metrics, Splits, expand_binary_class, split_integers, to_singleton_row

CHECKPOINT_EVERY = 100
MAX_EPOCHS = 50
BATCH_SIZE = 500

NOT_BOT = 0
BOT = 1
TRAINING_SPLIT = 0.5
VALIDATION_SPLIT = 0.2
TESTING_SPLIT = 0.3
TRAINER_PATIENCE = 10
SCHEDULER_PATIENCE = 3
METRIC_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
METRICS = ("accuracy", "precision", "recall")


def _make_splits(data: Dataset):
    length = len(data)
    split_lengths = split_integers(length, (TRAINING_SPLIT, VALIDATION_SPLIT, TESTING_SPLIT))

    splits = random_split(data, split_lengths)

    return Splits(data, *splits)


@pytest.fixture(scope="module")
def cresci_genuine_accounts_split(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    return _make_splits(cresci_genuine_accounts_tweets_tensors)


@pytest.fixture(scope="module")
def five38_split(five38_tweets_tensors: Five38TweetTensorDataset):
    return _make_splits(five38_tweets_tensors)


def _make_loader(notbot_splits: Splits, bot_splits: Splits, subset: str):
    notbot = SingleLabelDataset(getattr(notbot_splits, subset), NOT_BOT)
    bot = SingleLabelDataset(getattr(bot_splits, subset), BOT)

    dataset = ConcatDataset([notbot, bot])
    return DataLoader(dataset=dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=sentence_label_pad)


@pytest.fixture(scope="module")
def training_data(cresci_genuine_accounts_split: Splits, five38_split: Splits):
    return _make_loader(cresci_genuine_accounts_split, five38_split, "training")


@pytest.fixture(scope="module")
def validation_data(cresci_genuine_accounts_split: Splits, five38_split: Splits):
    return _make_loader(cresci_genuine_accounts_split, five38_split, "validation")


@pytest.fixture(scope="module")
def testing_data(cresci_genuine_accounts_split: Splits, five38_split: Splits):
    return _make_loader(cresci_genuine_accounts_split, five38_split, "testing")


def test_split_ratios_add_to_1():
    # Very specifically want these numbers to *equal* 1.0 here
    assert TRAINING_SPLIT + VALIDATION_SPLIT + TESTING_SPLIT == 1.0


@modes("cuda")
def test_cresci_genuine_accounts_split_add_up(cresci_genuine_accounts_split: Splits):
    total = len(cresci_genuine_accounts_split.full)
    training_split = len(cresci_genuine_accounts_split.training)
    validation_split = len(cresci_genuine_accounts_split.validation)
    testing_split = len(cresci_genuine_accounts_split.testing)

    assert training_split + validation_split + testing_split == total


@modes("cuda")
def test_bot_split_add_up(five38_split: Splits):
    total = len(five38_split.full)
    training_split = len(five38_split.training)
    validation_split = len(five38_split.validation)
    testing_split = len(five38_split.testing)

    assert training_split + validation_split + testing_split == total


@pytest.fixture(scope="module")
def model(mode: str, device: torch.device, glove_embedding: WordEmbeddings):
    lstm = ContextualLSTM(glove_embedding, device=device)

    if mode == 'dp':
        lstm = DataParallel(lstm)

    return lstm


@pytest.fixture(scope="module", params=[
    pytest.param((ASGD, {"lr": 0.1}), id="ASGD", marks=pytest.mark.ASGD),
    pytest.param((Adagrad, {"lr": 0.1}), id="Adagrad", marks=pytest.mark.Adagrad),
    pytest.param((Adadelta, {}), id="Adadelta", marks=pytest.mark.Adadelta),
    pytest.param((Adam, {}), id="Adam", marks=pytest.mark.Adam),
    pytest.param((Adam, {"lr": 0.01}), id="Adam(lr=0.01)", marks=pytest.mark.Adam),
    pytest.param((Adam, {"lr": 0.1}), id="Adam(lr=0.1)", marks=pytest.mark.Adam),
    pytest.param((SGD, {"lr": 0.1, "momentum": 0.9, "nesterov": True}), id="SGD(lr=0.1)", marks=pytest.mark.SGD),
    pytest.param((SGD, {"lr": 0.01, "momentum": 0.9, "nesterov": True}), id="SGD(lr=0.01)", marks=pytest.mark.SGD),
    pytest.param((RMSprop, {}), id="RMSprop(momentum=False)", marks=pytest.mark.RMSprop),
    pytest.param((RMSprop, {"momentum": 0.9}), id="RMSprop(momentum=True)", marks=pytest.mark.RMSprop),
    pytest.param((Rprop, {}), id="Rprop", marks=pytest.mark.Rprop),
])
def optimizer(request, model: Module):
    return request.param[0](model.parameters(), **request.param[1])


@pytest.fixture(scope="module")
def trainer(request, make_trainer, device: torch.device, optimizer: Optimizer, model: Module):
    return make_trainer(device, model, optimizer)


@pytest.fixture(scope="module")
def evaluator(trainer: Engine, device: torch.device):
    return ignite.engine.create_supervised_evaluator(
        trainer.state.model,
        metrics={
            "loss": Loss(trainer.state.criterion, output_transform=to_singleton_row),
            "accuracy": BinaryAccuracy(output_transform=to_singleton_row),
            "recall": Recall(average=True, output_transform=expand_binary_class),
            "precision": Precision(average=True, output_transform=expand_binary_class),
        }
    )


@pytest.fixture(scope="module")
def trained_model(request, trainer: Engine, optimizer, evaluator: Engine, training_data: DataLoader, validation_data: DataLoader):

    @trainer.on(Events.STARTED)
    def init_metrics(trainer: Engine):
        trainer.state.training_metrics = Metrics([], [], [], [])
        trainer.state.validation_metrics = Metrics([], [], [], [])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(trainer: Engine):
        training_metrics = evaluator.run(training_data).metrics  # type: Dict[str, float]
        trainer.state.training_metrics.loss.append(training_metrics["loss"])
        trainer.state.training_metrics.accuracy.append(training_metrics["accuracy"])
        trainer.state.training_metrics.recall.append(training_metrics["recall"])
        trainer.state.training_metrics.precision.append(training_metrics["precision"])

        validation_metrics = evaluator.run(validation_data).metrics  # type: Dict[str, float]
        trainer.state.validation_metrics.loss.append(validation_metrics["loss"])
        trainer.state.validation_metrics.accuracy.append(validation_metrics["accuracy"])
        trainer.state.validation_metrics.recall.append(validation_metrics["recall"])
        trainer.state.validation_metrics.precision.append(validation_metrics["precision"])
        scheduler.step(validation_metrics["loss"])

    timer = Timer(average=True)

    @trainer.on(Events.COMPLETED)
    def record_time(trainer: Engine):
        trainer.state.duration = timer.value()

    def score_function(trainer: Engine) -> float:
        return -trainer.state.validation_metrics.loss[-1]

    handler = EarlyStopping(patience=TRAINER_PATIENCE, score_function=score_function, trainer=trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    timer.attach(trainer, start=Events.STARTED, pause=Events.COMPLETED)

    trainer.run(training_data, max_epochs=MAX_EPOCHS)

    return trainer


@pytest.fixture(scope="module")
def testing_metrics(evaluator: Engine, testing_data: DataLoader):
    return evaluator.run(testing_data).metrics


@modes("cuda", "dp")
@report_metrics
def test_training_complete(trained_model: Engine):
    assert trained_model is not None

    return trained_model


@modes("cuda", "dp")
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("threshold", METRIC_THRESHOLDS)
def test_metrics_training_set(trained_model: Engine, threshold: float, metric: str):
    m = getattr(trained_model.state.training_metrics, metric)
    assert m[-1] >= threshold


@modes("cuda", "dp")
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("threshold", METRIC_THRESHOLDS)
def test_metrics_validation_set(trained_model: Engine, threshold: float, metric: str):
    m = getattr(trained_model.state.validation_metrics, metric)
    assert m[-1] >= threshold


@modes("cuda", "dp")
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("threshold", METRIC_THRESHOLDS)
def test_metrics_testing_set(testing_metrics: Dict[str, float], threshold: float, metric: str):
    assert testing_metrics[metric] >= threshold
