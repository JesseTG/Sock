from typing import Callable, Sequence, Tuple
from collections import namedtuple
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
from sockpuppet.model.dataset import LabelDataset, sentence_collate_batch
from tests.marks import needs_cuda, needs_cudnn

VALIDATE_EVERY = 100
CHECKPOINT_EVERY = 100
MAX_EPOCHS = 5
BATCH_SIZE = 8


torch.manual_seed(0)


def is_monotonically_decreasing(numbers: Sequence[float]) -> bool:
    # TODO: Is there a Pythonic one-liner here?
    for i in range(len(numbers) - 1):
        if numbers[i] <= numbers[i + 1]:
            return False

    return True


def make_data(device, max_index, total):
    num_words = max_index - 1
    size = (total // 2, 32)

    tensor0 = torch.randint(low=0, high=num_words // 2, size=size, dtype=torch.long, device=device)
    labels0 = torch.zeros([len(tensor0)], dtype=torch.float, device=device)
    # Zeros have lower-numbered word indices

    tensor1 = torch.randint(low=(num_words // 2) + 1, high=num_words, size=size, dtype=torch.long, device=device)
    labels1 = torch.ones([len(tensor1)], dtype=torch.float, device=device)
    # Ones have higher-numbered word indices

    return LabelDataset(torch.cat([tensor0, tensor1]), torch.cat([labels0, labels1]))


def sentence_collate(sentences: Sequence[Tuple[LongTensor, LongTensor]]) -> Tuple[LongTensor, LongTensor]:
    sentences = sorted(sentences, key=lambda x: len(x[0]), reverse=True)

    padded = pad_sequence([s[0] for s in sentences], False, 0)
    catted = torch.tensor([s[1] for s in sentences], dtype=torch.long, device=sentences[0][0].dtype)
    return (padded, catted)


@pytest.fixture(scope="module")
def training_dataset(device, glove_embedding: WordEmbeddings):
    return make_data(device, len(glove_embedding), 256)


@pytest.fixture(scope="module")
def validation_dataset(device, glove_embedding: WordEmbeddings):
    return make_data(device, len(glove_embedding), 1024)


@pytest.fixture(scope="module", params=[1, BATCH_SIZE])
def training_loader(request, training_dataset: LabelDataset):
    return DataLoader(training_dataset, batch_size=request.param)


@pytest.fixture(scope="module", params=[1, BATCH_SIZE])
def validation_loader(request, validation_dataset: Dataset):
    return DataLoader(validation_dataset, batch_size=request.param)


def test_training_runs(trainer: Engine, training_loader: DataLoader):

    result = trainer.run(training_loader, max_epochs=MAX_EPOCHS)

    assert result is not None


# @needs_cuda
# def test_training_cuda_faster_than_cpu(trainer_cpu: Engine, trainer_cuda: Engine, training_loader: DataLoader, training_loader_cuda: DataLoader):
#     start_cpu = time.time()
#     result_cpu = trainer_cpu.run(training_loader, max_epochs=MAX_EPOCHS)
#     duration_cpu = time.time() - start_cpu

#     start_cuda = time.time()
#     result_cuda = trainer_cuda.run(training_loader_cuda, max_epochs=MAX_EPOCHS)
#     duration_cuda = time.time() - start_cuda

#     assert duration_cuda < duration_cpu

# TODO: Ensure non-blocking CUDA works
# TODO: Ensure pinned memory works


def test_training_doesnt_change_word_embeddings(trainer: Engine, training_loader: DataLoader, glove_embedding: WordEmbeddings):
    embeddings = torch.tensor(glove_embedding.vectors)
    result = trainer.run(training_loader, max_epochs=MAX_EPOCHS)

    assert trainer.state.model.word_embeddings.vectors[0].cpu().numpy() == pytest.approx(embeddings[0].cpu().numpy())
    assert trainer.state.model.word_embeddings.vectors.data_ptr() != embeddings.data_ptr()


@pytest.mark.cuda_only  # CUDA only, to save time
def test_training_improves_metrics(device, trainer: Engine, training_loader: DataLoader, validation_loader: DataLoader):
    def tf(y):
        # TODO: Move to general utility function elsewhere
        return (y[0].reshape(-1, 1), y[1].reshape(-1, 1))

    mapping = torch.tensor([[1, 0], [0, 1]], device=device, dtype=torch.long)

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
        validator.run(validation_loader)
        trainer.state.loss.append(validator.state.metrics["loss"])
        trainer.state.accuracy.append(validator.state.metrics["accuracy"])
        trainer.state.recall.append(validator.state.metrics["recall"])
        trainer.state.precision.append(validator.state.metrics["precision"])

    trainer.run(training_loader, max_epochs=25)

    assert trainer.state.loss[0] > trainer.state.loss[-1]
    assert trainer.state.accuracy[0] < trainer.state.accuracy[-1]
    assert trainer.state.recall[0] < trainer.state.recall[-1]
    assert trainer.state.precision[0] < trainer.state.precision[-1]

    assert trainer.state.accuracy[-1] >= 0.25
    assert trainer.state.accuracy[-1] >= 0.45
