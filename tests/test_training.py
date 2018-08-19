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
from sockpuppet.model.nn.ContextualLSTM import ContextualLSTM
from sockpuppet.model.embedding import WordEmbeddings
from sockpuppet.model.dataset import LabelDataset, sentence_collate_batch
from tests.marks import *

BATCH_SIZES = [1, 8, 32, 64, 128]
VALIDATE_EVERY = 100
CHECKPOINT_EVERY = 100
MAX_EPOCHS = 5


torch.manual_seed(0)

DataLoaders = namedtuple("DataLoaders", ["training", "validation"])
TensorPair = namedtuple("TensorPair", ["data", "labels"])


def is_monotonically_decreasing(numbers: Sequence[float]) -> bool:
    # TODO: Is there a Pythonic one-liner here?
    for i in range(len(numbers) - 1):
        if numbers[i] <= numbers[i + 1]:
            return False

    return True


def make_data(device, max_index, total):
    # TODO: Make the data follow some pattern instead of random noise
    num_words = max_index - 1
    size = (total // 2, 32)

    tensor0 = torch.randint(low=0, high=num_words // 2, size=size, dtype=torch.long, device=device)
    labels0 = torch.zeros([len(tensor0)], dtype=torch.float, device=device)
    # Zeros have lower-numbered word indices

    tensor1 = torch.randint(low=(num_words // 2) + 1, high=num_words, size=size, dtype=torch.long, device=device)
    labels1 = torch.ones([len(tensor1)], dtype=torch.float, device=device)
    # Ones have higher-numbered word indices

    return TensorPair(torch.cat([tensor0, tensor1]), torch.cat([labels0, labels1]))

###############################################################################


@pytest.fixture(scope="module")
def training_tensors(request, device):
    return request.getfixturevalue(f"training_tensors_{device}")


@pytest.fixture(scope="module")
def training_tensors_cpu(glove_data: DataFrame):
    return make_data("cpu", len(glove_data), 256)


@pytest.fixture(scope="module")
def training_tensors_cuda(training_tensors_cpu: TensorPair):
    return (
        training_tensors_cpu[0].to("cuda", non_blocking=True),
        training_tensors_cpu[1].to("cuda", non_blocking=True),
    )


@pytest.fixture(scope="module")
def training_tensors_dp(training_tensors_cuda: TensorPair):
    return training_tensors_cuda

###############################################################################


###############################################################################
@pytest.fixture(scope="module")
def validation_tensors(request, device):
    return request.getfixturevalue(f"validation_tensors_{device}")


@pytest.fixture(scope="module")
def validation_tensors_cpu(glove_data: DataFrame):
    return make_data("cpu", len(glove_data), 1024)


@pytest.fixture(scope="module")
def validation_tensors_cuda(validation_tensors_cpu: TensorPair):
    return (
        validation_tensors_cpu[0].to("cuda", non_blocking=True),
        validation_tensors_cpu[1].to("cuda", non_blocking=True),
    )


@pytest.fixture(scope="module")
def validation_tensors_dp(validation_tensors_cuda: TensorPair):
    return validation_tensors_cuda

###############################################################################


@pytest.fixture(scope="module")
def training_dataset(training_tensors: TensorPair):
    return LabelDataset(*training_tensors)


@pytest.fixture(scope="module")
def validation_dataset(validation_tensors: TensorPair):
    return LabelDataset(*validation_tensors)


@pytest.fixture(scope="module", params=BATCH_SIZES)
def dataloaders(request, training_dataset: LabelDataset, validation_dataset: LabelDataset):
    return DataLoaders(
        DataLoader(training_dataset, batch_size=request.param),
        DataLoader(validation_dataset, batch_size=request.param),
    )


@record_runtime
@pytest.mark.dependency(name="test_training_runs")
def test_training_runs(request, trainer: Engine, dataloaders: DataLoaders):
    result = trainer.run(dataloaders.training, max_epochs=MAX_EPOCHS)

    assert result is not None


@needs_cuda
@pytest.mark.dependency(depends=["test_training_runs"])
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_training_cuda_faster_than_cpu(batch_size: int):
    duration_cpu = test_training_runs.duration[f"test_training_runs[cpu-{batch_size}]"]
    duration_cuda = test_training_runs.duration[f"test_training_runs[cuda-{batch_size}]"]

    assert duration_cuda < duration_cpu


# TODO: Ensure non-blocking CUDA works
# TODO: Ensure pinned memory works

# TODO: Split the training process off into a fixture


def test_training_doesnt_change_word_embeddings(trainer: Engine, dataloaders: DataLoaders, glove_embedding: WordEmbeddings):
    embeddings = torch.tensor(glove_embedding.vectors)
    result = trainer.run(dataloaders.training, max_epochs=MAX_EPOCHS)

    vectors = trainer.state.model.word_embeddings.vectors
    assert vectors.data_ptr() != embeddings.data_ptr()
    assert vectors[0].cpu().numpy() == pytest.approx(embeddings[0].cpu().numpy())


@devices("cuda")  # CUDA only, to save time
def test_training_improves_metrics(device, trainer: Engine, dataloaders: DataLoaders):
    def tf(y):
        # TODO: Move to general utility function elsewhere
        return (y[0].reshape(-1, 1), y[1].reshape(-1, 1))

    mapping = torch.as_tensor([[1, 0], [0, 1]], device=device, dtype=torch.long)

    def tf_2class(output):
        y_pred, y = output

        y_pred = mapping.index_select(0, y_pred.round().to(torch.long))

        # TODO: Recall metric isn't meant to be used for a binary class, so expand 0 to [1, 0] and 1 to [0, 1]
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
        validator.run(dataloaders.validation)
        trainer.state.loss.append(validator.state.metrics["loss"])
        trainer.state.accuracy.append(validator.state.metrics["accuracy"])
        trainer.state.recall.append(validator.state.metrics["recall"])
        trainer.state.precision.append(validator.state.metrics["precision"])

    trainer.run(dataloaders.training, max_epochs=25)

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
