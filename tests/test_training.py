from typing import Callable, Sequence, Tuple
import time

import pytest
import torch
import ignite

from torch import Tensor, LongTensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from ignite.engine import Events, Engine
import ignite.metrics
from sockpuppet.model.nn.ContextualLSTM import ContextualLSTM
from sockpuppet.model.embedding import WordEmbeddings
from sockpuppet.model.dataset import LabelDataset
from tests.marks import needs_cuda, needs_cudnn

VALIDATE_EVERY = 100
CHECKPOINT_EVERY = 100
MAX_EPOCHS = 10
BATCH_SIZE = 8


torch.manual_seed(0)


def is_monotonically_decreasing(numbers: Sequence[float]) -> bool:
    # TODO: Is there a Pythonic one-liner here?
    for i in range(len(numbers) - 1):
        if numbers[i] <= numbers[i + 1]:
            return False

    return True


def sentence_collate(sentences: Sequence[Tuple[LongTensor, LongTensor]]) -> Tuple[LongTensor, LongTensor]:
    sentences = sorted(sentences, key=lambda x: len(x[0]), reverse=True)

    padded = pad_sequence([s[0] for s in sentences], False, 0)
    catted = torch.tensor([s[1] for s in sentences])
    return (padded, catted)


@pytest.fixture(scope="module")
def training_dataset(glove_embedding: WordEmbeddings):
    num_words = len(glove_embedding) - 1

    tensor0 = torch.randint(low=0, high=num_words // 2, size=(128, 32), dtype=torch.long)
    labels0 = torch.zeros([len(tensor0)], dtype=torch.long)

    tensor1 = torch.randint(low=(num_words // 2) + 1, high=num_words, size=(128, 32), dtype=torch.long)
    labels1 = torch.ones([len(tensor1)], dtype=torch.long)
    return LabelDataset(torch.cat([tensor0, tensor1]), torch.cat([labels0, labels1]))


@pytest.fixture(scope="module")
def training_dataset_cuda(training_dataset: LabelDataset):
    data = training_dataset.data.cuda()
    labels = training_dataset.labels.cuda()
    return LabelDataset(data, labels)


@pytest.fixture(scope="module")
def validation_dataset(glove_embedding: WordEmbeddings):
    num_words = len(glove_embedding) - 1

    tensor0 = torch.randint(low=0, high=num_words // 2, size=(512, 32), dtype=torch.long)
    labels0 = torch.zeros([len(tensor0)], dtype=torch.long)

    tensor1 = torch.randint(low=(num_words // 2) + 1, high=num_words, size=(512, 32), dtype=torch.long)
    labels1 = torch.ones([len(tensor1)], dtype=torch.long)
    return LabelDataset(torch.cat([tensor0, tensor1]), torch.cat([labels0, labels1]))


@pytest.fixture(scope="module")
def validation_dataset_cuda(validation_dataset: LabelDataset):
    data = validation_dataset.data.cuda()
    labels = validation_dataset.labels.cuda()
    return LabelDataset(data, labels)


@pytest.fixture(scope="module")
def training_data(training_dataset: LabelDataset):
    return DataLoader(training_dataset, batch_size=1)


@pytest.fixture(scope="module")
def training_data_batched(training_dataset: LabelDataset):
    return DataLoader(training_dataset, batch_size=BATCH_SIZE)


@pytest.fixture(scope="module")
def training_data_cuda(training_dataset_cuda: LabelDataset):
    return DataLoader(training_dataset_cuda, batch_size=1)


@pytest.fixture(scope="module")
def training_data_cuda_batched(training_dataset_cuda: LabelDataset):
    return DataLoader(training_dataset_cuda, batch_size=BATCH_SIZE)


@pytest.fixture(scope="module")
def validation_data(validation_dataset: Dataset):
    return DataLoader(validation_dataset, batch_size=1)


@pytest.fixture(scope="module")
def validation_data_batched(validation_dataset: Dataset):
    return DataLoader(validation_dataset, batch_size=BATCH_SIZE)


@pytest.fixture(scope="module")
def validation_data_cuda(validation_dataset_cuda: LabelDataset):
    return DataLoader(validation_dataset_cuda, batch_size=1)


@pytest.fixture(scope="module")
def validation_data_cuda_batched(validation_dataset_cuda: Dataset):
    return DataLoader(validation_dataset_cuda, batch_size=BATCH_SIZE)


def test_training_runs_cpu(trainer_cpu: Engine, training_data: DataLoader):
    result = trainer_cpu.run(training_data, max_epochs=MAX_EPOCHS)

    assert result is not None


def test_training_runs_cpu_batched(trainer_cpu: Engine, training_data_batched: DataLoader):
    result = trainer_cpu.run(training_data_batched, max_epochs=MAX_EPOCHS)

    assert result is not None


@needs_cuda
def test_training_runs_cuda(trainer_cuda: Engine, training_data_cuda: DataLoader):
    result = trainer_cuda.run(training_data_cuda, max_epochs=MAX_EPOCHS)

    assert result is not None


@needs_cuda
def test_training_runs_cuda_batched(trainer_cuda: Engine, training_data_cuda_batched: DataLoader):
    result = trainer_cuda.run(training_data_cuda_batched, max_epochs=MAX_EPOCHS)

    assert result is not None


@needs_cuda
def test_training_cuda_faster_than_cpu(trainer_cpu: Engine, trainer_cuda: Engine, training_data: DataLoader, training_data_cuda: DataLoader):
    start_cpu = time.time()
    result_cpu = trainer_cpu.run(training_data, max_epochs=MAX_EPOCHS)
    duration_cpu = time.time() - start_cpu

    start_cuda = time.time()
    result_cuda = trainer_cuda.run(training_data_cuda, max_epochs=MAX_EPOCHS)
    duration_cuda = time.time() - start_cuda

    assert duration_cuda < duration_cpu

# TODO: Ensure non-blocking CUDA works
# TODO: Ensure pinned memory works


def test_training_doesnt_change_word_embeddings(trainer_cpu: Engine, training_data: DataLoader, glove_embedding: WordEmbeddings):
    embeddings = torch.tensor(glove_embedding.vectors)
    result = trainer_cpu.run(training_data, max_epochs=MAX_EPOCHS)

    assert trainer_cpu.state.model.word_embeddings.vectors.numpy()[0] == pytest.approx(embeddings.numpy()[0])
    assert trainer_cpu.state.model.word_embeddings.vectors.data_ptr() != embeddings.data_ptr()


@needs_cuda
def test_training_improves_metrics(trainer_cuda: Engine, training_data_cuda: DataLoader, validation_data_cuda: DataLoader):
    validator = ignite.engine.create_supervised_evaluator(
        trainer_cuda.state.model,
        metrics={
            "loss": ignite.metrics.Loss(trainer_cuda.state.criterion),
            "accuracy": ignite.metrics.CategoricalAccuracy(),
            "recall": ignite.metrics.Recall(average=True),
            "precision": ignite.metrics.Precision(average=True),
        }
    )

    @trainer_cuda.on(Events.STARTED)
    def init_metrics(trainer: Engine):
        trainer.state.loss = []
        trainer.state.accuracy = []
        trainer.state.recall = []
        trainer.state.precision = []

    @trainer_cuda.on(Events.EPOCH_COMPLETED)
    def validate(trainer: Engine):
        validator.run(validation_data_cuda)
        trainer.state.loss.append(validator.state.metrics["loss"])
        trainer.state.accuracy.append(validator.state.metrics["accuracy"])
        trainer.state.recall.append(validator.state.metrics["recall"])
        trainer.state.precision.append(validator.state.metrics["precision"])

    trainer_cuda.run(training_data_cuda, max_epochs=50)

    assert is_monotonically_decreasing(trainer_cuda.state.loss)
    assert trainer_cuda.state.accuracy[0] < trainer_cuda.state.accuracy[-1]
    assert trainer_cuda.state.recall[0] < trainer_cuda.state.recall[-1]
    assert trainer_cuda.state.precision[0] < trainer_cuda.state.precision[-1]

    assert trainer_cuda.state.accuracy[-1] >= 0.25
    assert trainer_cuda.state.accuracy[-1] >= 0.45
