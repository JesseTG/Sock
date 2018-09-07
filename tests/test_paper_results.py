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
from sockpuppet.model.dataset import CresciTensorTweetDataset, NbcTweetTensorDataset, Five38TweetTensorDataset
from sockpuppet.model.data import sentence_pad, sentence_label_pad, WordEmbeddings
from sockpuppet.utils import split_integers, expand_binary_class, to_singleton_row, Splits, Metrics
from tests.marks import *

CHECKPOINT_EVERY = 100
MAX_EPOCHS = 5
BATCH_SIZE = 1000

NOT_BOT = 0
BOT = 1
TRAINING_SPLIT = 0.4
VALIDATION_SPLIT = 0.1
TESTING_SPLIT = 0.5
TRAINER_PATIENCE = 100
METRIC_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
METRICS = ("accuracy", "precision", "recall")



@pytest.fixture(scope="module")
def cresci_genuine_accounts_split(cresci_genuine_accounts_tweets_tensors_cpu: CresciTensorTweetDataset):
    length = len(cresci_genuine_accounts_tweets_tensors_cpu)
    split_lengths = split_integers(length, (TRAINING_SPLIT, VALIDATION_SPLIT, TESTING_SPLIT))

    splits = random_split(cresci_genuine_accounts_tweets_tensors_cpu, split_lengths)

    return Splits(cresci_genuine_accounts_tweets_tensors_cpu, *splits)


@pytest.fixture(scope="module")
def cresci_social_spambots_1_split(cresci_social_spambots_1_tweets_tensors_cpu: CresciTensorTweetDataset):
    length = len(cresci_social_spambots_1_tweets_tensors_cpu)
    split_lengths = split_integers(length, (TRAINING_SPLIT, VALIDATION_SPLIT, TESTING_SPLIT))

    splits = random_split(cresci_social_spambots_1_tweets_tensors_cpu, split_lengths)

    return Splits(cresci_social_spambots_1_tweets_tensors_cpu, *splits)


@pytest.fixture(scope="module")
def training_data(
    cresci_genuine_accounts_split: Splits,
    cresci_social_spambots_1_split: Splits
):
    notbot = SingleLabelDataset(cresci_genuine_accounts_split.training, NOT_BOT)
    bot = SingleLabelDataset(cresci_social_spambots_1_split.training, BOT)

    dataset = ConcatDataset([notbot, bot])
    return DataLoader(dataset=dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=sentence_label_pad)


@pytest.fixture(scope="module")
def validation_data(
    cresci_genuine_accounts_split: Splits,
    cresci_social_spambots_1_split: Splits
):
    notbot = SingleLabelDataset(cresci_genuine_accounts_split.validation, NOT_BOT)
    bot = SingleLabelDataset(cresci_social_spambots_1_split.validation, BOT)

    dataset = ConcatDataset([notbot, bot])
    return DataLoader(dataset=dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=sentence_label_pad)


@pytest.fixture(scope="module")
def testing_data(
    cresci_genuine_accounts_split: Splits,
    cresci_social_spambots_1_split: Splits
):
    notbot = SingleLabelDataset(cresci_genuine_accounts_split.testing, NOT_BOT)
    bot = SingleLabelDataset(cresci_social_spambots_1_split.testing, BOT)

    dataset = ConcatDataset([notbot, bot])
    return DataLoader(dataset=dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=sentence_label_pad)


def test_split_ratios_add_to_1():
    # Very specifically want these numbers to *equal* 1.0 here
    assert TRAINING_SPLIT + VALIDATION_SPLIT + TESTING_SPLIT == 1.0

# TODO: Document that this section is *strictly* for using the Cresci paper's data


@modes("cuda")
def test_cresci_genuine_accounts_split_add_up(cresci_genuine_accounts_split: Splits):
    total = len(cresci_genuine_accounts_split.full)
    training_split = len(cresci_genuine_accounts_split.training)
    validation_split = len(cresci_genuine_accounts_split.validation)
    testing_split = len(cresci_genuine_accounts_split.testing)

    assert training_split + validation_split + testing_split == total


@modes("cuda")
def test_cresci_social_spambots_1_split_add_up(cresci_social_spambots_1_split: Splits):
    total = len(cresci_social_spambots_1_split.full)
    training_split = len(cresci_social_spambots_1_split.training)
    validation_split = len(cresci_social_spambots_1_split.validation)
    testing_split = len(cresci_social_spambots_1_split.testing)

    assert training_split + validation_split + testing_split == total


@pytest.fixture(scope="module")
def model(mode: str, device: torch.device, glove_embedding: WordEmbeddings):
    lstm = ContextualLSTM(glove_embedding, device=device)

    if mode == 'dp':
        lstm = DataParallel(lstm)

    return lstm


@pytest.fixture(scope="module", params=[
    pytest.param((ASGD, {"lr": 0.1}), id="ASGD"),
    pytest.param((Adagrad, {"lr": 0.1}), id="Adagrad"),
    pytest.param((Adadelta, {}), id="Adadelta"),
    pytest.param((Adam, {}), id="Adam"),
    pytest.param((Adam, {"lr": 0.01}), id="Adam(lr=0.01)"),
    pytest.param((Adam, {"lr": 0.1}), id="Adam(lr=0.1)"),
    pytest.param((SGD, {"lr": 0.1, "momentum": 0.9, "nesterov": True}), id="SGD"),
    pytest.param((RMSprop, {}), id="RMSprop-no-momentum"),
    pytest.param((RMSprop, {"momentum": 0.9}), id="RMSprop-momentum"),
    pytest.param((Rprop, {}), id="Rprop"),
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
def trained_model(request, trainer: Engine, evaluator: Engine, training_data: DataLoader, validation_data: DataLoader):

    @trainer.on(Events.STARTED)
    def init_metrics(trainer: Engine):
        trainer.state.training_metrics = Metrics([], [], [], [])
        trainer.state.validation_metrics = Metrics([], [], [], [])

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
