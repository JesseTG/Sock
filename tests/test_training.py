from typing import Callable, Sequence, Tuple
from collections import namedtuple
import time

import pytest
import torch
import ignite
from pandas import DataFrame

from torch import Tensor, LongTensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from ignite.engine import Events, Engine
import ignite.metrics
from sockpuppet.model.nn import ContextualLSTM
from sockpuppet.model.data import WordEmbeddings, sentence_label_pad
from sockpuppet.model.dataset import LabelDataset
from tests.marks import *

BATCH_SIZES = [100, 250, 500, 1000]
VALIDATE_EVERY = 100
CHECKPOINT_EVERY = 100
MAX_EPOCHS = 5
NUM_GPUS = torch.cuda.device_count()

torch.manual_seed(0)

DataLoaders = namedtuple("DataLoaders", ["training", "validation"])
Datasets = namedtuple("Datasets", ["training", "validation"])
TensorPair = namedtuple("TensorPair", ["data", "labels"])


def is_monotonically_decreasing(numbers: Sequence[float]) -> bool:
    # TODO: Is there a Pythonic one-liner here?
    for i in range(len(numbers) - 1):
        if numbers[i] <= numbers[i + 1]:
            return False

    return True


def make_data(device: torch.device, max_index: int, total: int):
    # TODO: Make the data follow some pattern instead of random noise (e.g. a normal distribution over the range of indices)
    # use torch.randn

    size = (total // 2, 32)
    std = round(max_index * 0.001)

    mean0 = (max_index // 2) // 2 + 1
    tensor0 = torch.normal(torch.full(size, mean0, dtype=torch.float, device=device), std=std).round().long()
    labels0 = torch.zeros([len(tensor0)], dtype=torch.float, device=device)
    # Zeros have lower-numbered word indices

    mean1 = ((max_index // 2) + max_index) // 2 + 1
    tensor1 = torch.normal(torch.full(size, mean1, dtype=torch.float, device=device), std=std).round().long()
    labels1 = torch.ones([len(tensor1)], dtype=torch.float, device=device)
    # Ones have higher-numbered word indices

    return TensorPair(torch.cat([tensor0, tensor1]), torch.cat([labels0, labels1]))


###############################################################################
# training_tensors ############################################################

@pytest.fixture(scope="module")
def training_tensors(request, device: torch.device):
    return request.getfixturevalue(f"training_tensors_{device.type}")


@pytest.fixture(scope="module")
def training_tensors_cpu(glove_data: DataFrame):
    return make_data("cpu", len(glove_data), 8000)


@pytest.fixture(scope="module")
def training_tensors_cuda(training_tensors_cpu: TensorPair):
    return (
        training_tensors_cpu[0].to("cuda", non_blocking=True),
        training_tensors_cpu[1].to("cuda", non_blocking=True),
    )

###############################################################################


###############################################################################
# validation_tensors ##########################################################

@pytest.fixture(scope="module")
def validation_tensors(request, device: torch.device):
    return request.getfixturevalue(f"validation_tensors_{device.type}")


@pytest.fixture(scope="module")
def validation_tensors_cpu(glove_data: DataFrame):
    return make_data("cpu", len(glove_data), 32000)


@pytest.fixture(scope="module")
def validation_tensors_cuda(validation_tensors_cpu: TensorPair):
    return (
        validation_tensors_cpu[0].to("cuda", non_blocking=True),
        validation_tensors_cpu[1].to("cuda", non_blocking=True),
    )

###############################################################################


@pytest.fixture(scope="function")
def trainer(make_trainer, device: torch.device, lstm: ContextualLSTM):
    optimizer = torch.optim.SGD(
        lstm.parameters(),
        lr=0.1,
        momentum=0.9,
        nesterov=True
    )
    return make_trainer(device, lstm, optimizer)


@pytest.fixture(scope="module")
def training_dataset(training_tensors: TensorPair):
    return LabelDataset(*training_tensors)


@pytest.fixture(scope="module")
def validation_dataset(validation_tensors: TensorPair):
    return LabelDataset(*validation_tensors)


@pytest.fixture(scope="module")
def datasets(training_dataset: LabelDataset, validation_dataset: LabelDataset):
    return Datasets(training_dataset, validation_dataset)


@pytest.fixture(scope="module", params=BATCH_SIZES)
def dataloaders(request, datasets: Datasets):
    return DataLoaders(
        DataLoader(datasets.training, batch_size=request.param, collate_fn=sentence_label_pad),
        DataLoader(datasets.validation, batch_size=request.param, collate_fn=sentence_label_pad),
    )


@modes("cpu", "cuda")
@pytest.mark.benchmark(group="make_data")
def test_bench_make_data(benchmark, device: torch.device, glove_data: DataFrame):
    result = benchmark(make_data, device, len(glove_data), 1024)

    assert result is not None


@modes("cuda")
@slow
@pytest.mark.benchmark(group="training")
def test_bench_training_cuda(benchmark, trainer: Engine, dataloaders: DataLoaders):
    result = benchmark(trainer.run, dataloaders.training, max_epochs=MAX_EPOCHS)

    assert result is not None


@modes("dp")
@slow
@pytest.mark.benchmark(group="training")
@pytest.mark.parametrize("batch_size", [b * NUM_GPUS for b in BATCH_SIZES])
def test_bench_training_dp(benchmark, trainer: Engine, training_dataset: LabelDataset, batch_size: int):
    loader = DataLoader(training_dataset, batch_size=batch_size, collate_fn=sentence_label_pad)
    result = benchmark(trainer.run, loader, max_epochs=MAX_EPOCHS)

    assert result is not None


# TODO: Split the training process off into a fixture

@modes("cuda")
def test_training_doesnt_change_word_embeddings(trainer: Engine, training_dataset: Dataset, glove_embedding: WordEmbeddings):
    # TODO: Make a dp version
    training = DataLoader(training_dataset, batch_size=1000, collate_fn=sentence_label_pad)
    embeddings = torch.tensor(glove_embedding.vectors)
    result = trainer.run(training, max_epochs=MAX_EPOCHS)

    vectors = trainer.state.model.word_embeddings.vectors
    assert vectors.data_ptr() != embeddings.data_ptr()
    assert vectors[0].cpu().numpy() == pytest.approx(embeddings[0].cpu().numpy())


@modes("cuda", "dp")  # CUDA only, to save time
@slow
def test_training_improves_metrics(device: torch.device, trainer: Engine, datasets: Datasets):
    def tf(y):
        # TODO: Move to general utility function elsewhere
        return (y[0].reshape(-1, 1), y[1].reshape(-1, 1))

    loaders = DataLoaders(
        DataLoader(datasets.training, batch_size=1000, collate_fn=sentence_label_pad),
        DataLoader(datasets.validation, batch_size=1000, collate_fn=sentence_label_pad),
    )
    mapping = torch.as_tensor([[1, 0], [0, 1]], device=device, dtype=torch.long)

    def tf_2class(output):
        y_pred, y = output

        y_pred = mapping.index_select(0, y_pred.round().to(torch.long))

        return (y_pred, y.to(torch.long))

    validator = ignite.engine.create_supervised_evaluator(
        trainer.state.model,
        metrics={
            "loss": ignite.metrics.Loss(trainer.state.criterion, output_transform=tf),
            "accuracy": ignite.metrics.BinaryAccuracy(output_transform=tf),
            "recall": ignite.metrics.Recall(average=True, output_transform=tf_2class),
            "precision": ignite.metrics.Precision(average=True, output_transform=tf_2class),
        }
    )

    @trainer.on(Events.STARTED)
    def init_metrics(trainer: Engine):
        trainer.state.loss = []
        trainer.state.accuracy = []
        trainer.state.recall = []
        trainer.state.precision = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(trainer: Engine):
        validator.run(loaders.validation)
        trainer.state.loss.append(validator.state.metrics["loss"])
        trainer.state.accuracy.append(validator.state.metrics["accuracy"])
        trainer.state.recall.append(validator.state.metrics["recall"])
        trainer.state.precision.append(validator.state.metrics["precision"])

    trainer.run(loaders.training, max_epochs=25)

    assert len(trainer.state.loss) > 1
    assert len(trainer.state.accuracy) > 1
    assert len(trainer.state.recall) > 1
    assert len(trainer.state.precision) > 1

    assert trainer.state.loss[0] > trainer.state.loss[-1]
    assert trainer.state.accuracy[0] < trainer.state.accuracy[-1]
    assert trainer.state.recall[0] < trainer.state.recall[-1]
    assert trainer.state.precision[0] < trainer.state.precision[-1]

    assert trainer.state.accuracy[-1] >= 0.25
    assert trainer.state.accuracy[-1] >= 0.45
