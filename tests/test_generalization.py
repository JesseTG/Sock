from collections import namedtuple
from typing import Callable, Sequence, Dict
import time

import pytest
import torch
import ignite

from torch import Tensor, LongTensor
from torch.nn import Module, DataParallel
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset, RandomSampler, random_split
from ignite.engine import Events, Engine, State
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss, BinaryAccuracy, Precision, Recall
from sockpuppet.model.nn import ContextualLSTM
from sockpuppet.model.dataset.label import LabelDataset, SingleLabelDataset
from sockpuppet.model.dataset import CresciTensorTweetDataset, Five38TweetTensorDataset
from sockpuppet.model.data import sentence_pad, sentence_label_pad, WordEmbeddings
from sockpuppet.utils import split_integers
from tests.marks import *

CHECKPOINT_EVERY = 100
MAX_EPOCHS = 10
BATCH_SIZE = 1000

NOT_BOT = 0
BOT = 1
TRAINING_SPLIT = 0.4
VALIDATION_SPLIT = 0.1
TESTING_SPLIT = 0.5
TRAINER_PATIENCE = 100
METRIC_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
METRICS = ("accuracy", "precision", "recall")

Splits = namedtuple("Splits", ("full", "training", "validation", "testing"))
Metrics = namedtuple("Metrics", ("accuracy", "loss", "precision", "recall"))


@pytest.fixture(scope="module")
def cresci_genuine_accounts_split(cresci_genuine_accounts_tweets_tensors: CresciTensorTweetDataset):
    length = len(cresci_genuine_accounts_tweets_tensors)
    split_lengths = split_integers(length, (TRAINING_SPLIT, VALIDATION_SPLIT, TESTING_SPLIT))

    splits = random_split(cresci_genuine_accounts_tweets_tensors, split_lengths)

    return Splits(cresci_genuine_accounts_tweets_tensors, *splits)


@pytest.fixture(scope="module")
def five38_split(five38_tweets_tensors: Five38TweetTensorDataset):
    length = len(five38_tweets_tensors)
    split_lengths = split_integers(length, (TRAINING_SPLIT, VALIDATION_SPLIT, TESTING_SPLIT))

    splits = random_split(five38_tweets_tensors, split_lengths)

    return Splits(five38_tweets_tensors, *splits)


@pytest.fixture(scope="module")
def training_data(
    cresci_genuine_accounts_split: Splits,
    five38_split: Splits
):
    notbot = SingleLabelDataset(cresci_genuine_accounts_split.training, NOT_BOT)
    bot = SingleLabelDataset(five38_split.training, BOT)

    dataset = ConcatDataset([notbot, bot])
    sampler = RandomSampler(dataset)
    return DataLoader(dataset=dataset, sampler=sampler, batch_size=BATCH_SIZE, collate_fn=sentence_label_pad)


@pytest.fixture(scope="module")
def validation_data(
    cresci_genuine_accounts_split: Splits,
    five38_split: Splits
):
    notbot = SingleLabelDataset(cresci_genuine_accounts_split.validation, NOT_BOT)
    bot = SingleLabelDataset(five38_split.validation, BOT)

    dataset = ConcatDataset([notbot, bot])
    sampler = RandomSampler(dataset)
    return DataLoader(dataset=dataset, sampler=sampler, batch_size=BATCH_SIZE, collate_fn=sentence_label_pad)


@pytest.fixture(scope="module")
def testing_data(
    cresci_genuine_accounts_split: Splits,
    five38_split: Splits
):
    notbot = SingleLabelDataset(cresci_genuine_accounts_split.testing, NOT_BOT)
    bot = SingleLabelDataset(five38_split.testing, BOT)

    dataset = ConcatDataset([notbot, bot])
    sampler = RandomSampler(dataset)
    return DataLoader(dataset=dataset, sampler=sampler, batch_size=BATCH_SIZE, collate_fn=sentence_label_pad)


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
def test_538_split_add_up(five38_split: Splits):
    total = len(five38_split.full)
    training_split = len(five38_split.training)
    validation_split = len(five38_split.validation)
    testing_split = len(five38_split.testing)

    assert training_split + validation_split + testing_split == total


@pytest.fixture(scope="module")
def trainer_engine(make_trainer, device: torch.device, mode: str, glove_embedding: WordEmbeddings):
    # I can't be bothered to figure out how fixture overriding works

    lstm = ContextualLSTM(glove_embedding, device=device)
    if mode == 'dp':
        return make_trainer(device, DataParallel(lstm))
    else:
        return make_trainer(device, lstm)


@pytest.fixture(scope="module")
def evaluator(trainer_engine: Engine, device: torch.device):
    def tf(y):
        # TODO: Move to general utility function elsewhere
        return (y[0].reshape(-1, 1), y[1].reshape(-1, 1))

    mapping = torch.tensor([[1, 0], [0, 1]], device=device, dtype=torch.long)

    def tf_2class(output):
        y_pred, y = output

        y_pred = mapping.index_select(0, y_pred.round().to(torch.long))

        return (y_pred, y.to(torch.long))

    return ignite.engine.create_supervised_evaluator(
        trainer_engine.state.model,
        metrics={
            "loss": Loss(trainer_engine.state.criterion, output_transform=tf),
            "accuracy": BinaryAccuracy(output_transform=tf),
            "recall": Recall(average=True, output_transform=tf_2class),
            "precision": Precision(average=True, output_transform=tf_2class),
        }
    )


@pytest.fixture(scope="module")
def trained_model(trainer_engine: Engine, evaluator: Engine, training_data: DataLoader, validation_data: DataLoader):

    @trainer_engine.on(Events.STARTED)
    def init_metrics(trainer_engine: Engine):
        trainer_engine.state.training_metrics = Metrics([], [], [], [])
        trainer_engine.state.validation_metrics = Metrics([], [], [], [])

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def validate(trainer_engine: Engine):
        training_metrics = evaluator.run(training_data).metrics  # type: Dict[str, float]
        trainer_engine.state.training_metrics.loss.append(training_metrics["loss"])
        trainer_engine.state.training_metrics.accuracy.append(training_metrics["accuracy"])
        trainer_engine.state.training_metrics.recall.append(training_metrics["recall"])
        trainer_engine.state.training_metrics.precision.append(training_metrics["precision"])

        validation_metrics = evaluator.run(validation_data).metrics  # type: Dict[str, float]
        trainer_engine.state.validation_metrics.loss.append(validation_metrics["loss"])
        trainer_engine.state.validation_metrics.accuracy.append(validation_metrics["accuracy"])
        trainer_engine.state.validation_metrics.recall.append(validation_metrics["recall"])
        trainer_engine.state.validation_metrics.precision.append(validation_metrics["precision"])

    def score_function(trainer_engine: Engine) -> float:
        return -trainer_engine.state.validation_metrics.loss[-1]

    handler = EarlyStopping(patience=TRAINER_PATIENCE, score_function=score_function, trainer=trainer_engine)
    trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, handler)

    trainer_engine.run(training_data, max_epochs=MAX_EPOCHS)

    return trainer_engine


@pytest.fixture(scope="module")
def testing_metrics(evaluator: Engine, testing_data: DataLoader):
    return evaluator.run(testing_data).metrics


# @pytest.fixture(scope="module")
# def nbc_metrics(evaluator: Engine, nbc_tweets_tensors: NbcTweetTensorDataset):
#     nbc_data = SingleLabelDataset(nbc_tweets_tensors, BOT)
#     nbc_loader = DataLoader(dataset=nbc_data, batch_size=BATCH_SIZE, collate_fn=sentence_label_pad)
#     return evaluator.run(nbc_loader).metrics


# @pytest.fixture(scope="module")
# def five38_metrics(evaluator: Engine, five38_tweets_tensors: Five38TweetTensorDataset):
#     five38_data = SingleLabelDataset(five38_tweets_tensors, BOT)
#     five38_loader = DataLoader(dataset=five38_data, batch_size=BATCH_SIZE, collate_fn=sentence_label_pad)
#     return evaluator.run(five38_loader).metrics


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


# @modes("cuda", "dp")
# @pytest.mark.parametrize("metric", METRICS)
# @pytest.mark.parametrize("threshold", METRIC_THRESHOLDS)
# def test_metrics_nbc(nbc_metrics: Dict[str, float], threshold: float, metric: str):
#     assert nbc_metrics[metric] >= threshold


# @modes("cuda", "dp")
# @pytest.mark.parametrize("metric", METRICS)
# @pytest.mark.parametrize("threshold", METRIC_THRESHOLDS)
# def test_metrics_538(five38_metrics: dict, threshold: float, metric: str):
#     assert five38_metrics[metric] >= threshold
