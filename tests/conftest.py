# -*- coding: utf-8 -*-
"""Defines fixtures available to all tests."""

import csv
import sys
import time
from typing import Callable, Dict, Sequence, Union

import cpuinfo
import ignite
import pandas
import pytest
import torch
import torch.multiprocessing
from ignite.engine import Engine, Events
from pandas import DataFrame
from pytest import Item, Session
from torch.nn import DataParallel, Module
from torch.optim import Optimizer

from sock.model.data import WordEmbeddings, tokenize
from sock.model.dataset.cresci import CresciTensorTweetDataset, CresciTweetDataset, CresciUserDataset
from sock.model.dataset.five38 import Five38TweetDataset, Five38TweetTensorDataset
from sock.model.dataset.nbc import NbcTweetDataset, NbcTweetTensorDataset
from sock.model.nn import ContextualLSTM
from sock.settings import TestConfig

from .marks import *

FIVE38_TWEET_PATH = f"{TestConfig.TRAINING_DATA_PATH}/538/tweets.csv"
NBC_TWEET_PATH = f"{TestConfig.TRAINING_DATA_PATH}/nbc/tweets.csv"
CRESCI_PATH = f"{TestConfig.TRAINING_DATA_PATH}/cresci-2017/datasets_full.csv"
GENUINE_ACCOUNT_TWEET_PATH = f"{CRESCI_PATH}/genuine_accounts.csv/tweets.csv"
GENUINE_ACCOUNT_USER_PATH = f"{CRESCI_PATH}/genuine_accounts.csv/users.csv"
SOCIAL_SPAMBOTS_1_TWEET_PATH = f"{CRESCI_PATH}/social_spambots_1.csv/tweets.csv"
SOCIAL_SPAMBOTS_1_USER_PATH = f"{CRESCI_PATH}/social_spambots_1.csv/users.csv"
GLOVE_PATH = f"{TestConfig.TRAINING_DATA_PATH}/glove/glove.twitter.27B.25d.txt"


def pytest_report_header(config, startdir):
    cpu = cpuinfo.get_cpu_info()  # type: Dict[str, Union[str, int, Sequence[str]]]

    return (
        f"omp_threads: {torch.get_num_threads()}, cuda_devices: {torch.cuda.device_count()}",
        f"default_dtype: {torch.get_default_dtype()}",
        f"arch: {cpu['arch']}, cores: {cpu['count']}",
        f"SOCK_TRAINING_DATA_PATH: {TestConfig.TRAINING_DATA_PATH}",
    )


def pytest_collection_modifyitems(session, config, items: Sequence[Item]):
    to_remove = set()
    for item in items:
        modes = item.get_closest_marker("modes")
        if modes is not None:
            # If we're explicitly using a subset of modes...
            if hasattr(item, "callspec") and hasattr(item.callspec, "params"):
                params = item.callspec.params
                if "mode" in params and params["mode"] not in modes.args:
                    # If this mode isn't in the list...
                    to_remove.add(item)

    for i in to_remove:
        items.remove(i)


def pytest_sessionstart(session: Session):
    try:
        if torch.cuda.is_available() and "spawn" in torch.multiprocessing.get_all_start_methods():
            torch.multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        print(e)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == 'call' and item.get_closest_marker("report_metrics") is not None:
        if "trained_model" in item.funcargs:
            report.user_properties.append(("trained_model", item.funcargs["trained_model"]))


def pytest_terminal_summary(terminalreporter, exitstatus):
    report_metrics = [r for r in terminalreporter.getreports(
        "") if "report_metrics" in r.keywords and r.when == 'setup']
    terminalreporter.section("Training Results")

    for report in report_metrics:
        engine = report.user_properties[0][1]  # type: Engine
        state = engine.state
        train = state.training_metrics
        valid = state.validation_metrics
        terminalreporter.line(report.nodeid)
        terminalreporter.line(f"{state.epoch} / {state.max_epochs} epochs, {state.iteration} iterations")
        terminalreporter.line(f"Time:\t{state.duration:.3f}s")

        if train.accuracy and valid.accuracy:
            terminalreporter.line(f"Accuracy:\ttrain={train.accuracy[-1]:.5f}, valid={valid.accuracy[-1]:.5f}")

        if train.precision and valid.precision:
            terminalreporter.line(f"Precision:\ttrain={train.precision[-1]:.5f}, valid={valid.precision[-1]:.5f}")

        if train.recall and valid.recall:
            terminalreporter.line(f"Recall:\ttrain={train.recall[-1]:.5f}, valid={valid.recall[-1]:.5f}")

        if train.loss and valid.loss:
            terminalreporter.line(f"Loss:\ttrain={train.loss[-1]:.5f}, valid={valid.loss[-1]:.5f}")

        terminalreporter.line("")


@pytest.fixture(scope="session", params=[
    pytest.param("cpu", marks=[cpu]),
    pytest.param("cuda", marks=[cuda, needs_cuda, needs_cudnn]),
    pytest.param("dp", marks=[dp, needs_cuda, needs_cudnn, needs_multiple_gpus])
])
def mode(request):
    '''
    Determines whether a test or fixture will be run on the CPU, on one CUDA device,
    or on all CUDA devices.
    '''
    return request.param


@pytest.fixture(scope="session")
def device(mode):
    '''
    Same as `mode`, but returns `"cuda"` instead of `"dp"`. You still need
    to explicitly filter out `"dp"` with `@modes`.
    '''
    if mode in ("cuda", "dp"):
        return torch.device("cuda", 0)
    elif mode == "cpu":
        return torch.device("cpu")

    return torch.device(mode)
    # TODO: Make this and mode entirely separate fixtures with no dependencies on each other,
    # but ensure in the collection hook that "mode" and "device" are consistent with one another


@pytest.fixture(scope="session")
def glove_data():
    return pandas.read_table(
        GLOVE_PATH,
        delim_whitespace=True,
        header=None,
        engine="c",
        encoding="utf8",
        na_filter=False,
        memory_map=True,
        quoting=csv.QUOTE_NONE
    )


@pytest.fixture(scope="session")
def glove_embedding(request, device: torch.device, glove_embedding_cpu: WordEmbeddings, glove_embedding_cuda: WordEmbeddings):
    """Load the GloVe embeddings."""
    return request.getfixturevalue(f"glove_embedding_{device.type}")


@pytest.fixture(scope="session")
def glove_embedding_cpu(glove_data: DataFrame):
    """Load the GloVe embeddings onto CPU memory."""
    return WordEmbeddings(glove_data, "cpu")


@pytest.fixture(scope="session")
def glove_embedding_cuda(glove_data: DataFrame):
    """Load the GloVe embeddings onto CUDA memory."""
    return WordEmbeddings(glove_data, "cuda")


@pytest.fixture(scope="session")
def glove_embedding_dp(glove_embedding_cuda: WordEmbeddings):
    """Load the GloVe embeddings onto CUDA memory."""
    return glove_embedding_cuda


@pytest.fixture(scope="function")
def lstm(request, mode: str):
    '''Creates a ContextualLSTM of either CPU or CUDA type'''
    return request.getfixturevalue(f"lstm_{mode}")


@pytest.fixture(scope="function")
def lstm_cpu(glove_embedding_cpu: WordEmbeddings):
    '''Creates a ContextualLSTM of CPU type'''
    return ContextualLSTM(glove_embedding_cpu, device="cpu")


@pytest.fixture(scope="function")
def lstm_cuda(glove_embedding_cuda: WordEmbeddings):
    '''Creates a ContextualLSTM of CUDA type'''
    return ContextualLSTM(glove_embedding_cuda, device="cuda")


@pytest.fixture(scope="function")
def lstm_dp(lstm_cuda: ContextualLSTM):
    '''Creates a ContextualLSTM of CUDA type'''
    return DataParallel(lstm_cuda)


@pytest.fixture(scope="session")
def cresci_genuine_accounts_tweets():
    """Load genuine_accounts/tweets.csv from cresci-2017"""
    return CresciTweetDataset(GENUINE_ACCOUNT_TWEET_PATH)

###############################################################################
# NBC tweets
###############################################################################


@pytest.fixture(scope="session")
def nbc_tweets():
    """Load the tweets revealed by NBC"""
    return NbcTweetDataset(NBC_TWEET_PATH)


@pytest.fixture(scope="session")
def nbc_tweets_tensors(request, device: torch.device):
    return request.getfixturevalue(f"nbc_tweets_tensors_{device.type}")


@pytest.fixture(scope="session")
def nbc_tweets_tensors_cpu(nbc_tweets: NbcTweetDataset, glove_embedding_cpu: WordEmbeddings):
    return NbcTweetTensorDataset(
        data_source=nbc_tweets,
        embeddings=glove_embedding_cpu,
        tokenizer=tokenize
    )


@pytest.fixture(scope="session")
def nbc_tweets_tensors_cuda(nbc_tweets: NbcTweetDataset, glove_embedding_cuda: WordEmbeddings):
    return NbcTweetTensorDataset(
        data_source=nbc_tweets,
        embeddings=glove_embedding_cuda,
        tokenizer=tokenize
    )

###############################################################################

###############################################################################
# 538 tweets
###############################################################################


@pytest.fixture(scope="session")
def five38_tweets():
    """Load the tweets revealed by 538"""
    return Five38TweetDataset(FIVE38_TWEET_PATH)


@pytest.fixture(scope="session")
def five38_tweets_tensors(request, device: torch.device):
    return request.getfixturevalue(f"five38_tweets_tensors_{device.type}")


@pytest.fixture(scope="session")
def five38_tweets_tensors_cpu(five38_tweets: Five38TweetDataset, glove_embedding_cpu: WordEmbeddings):
    return Five38TweetTensorDataset(
        data_source=five38_tweets,
        embeddings=glove_embedding_cpu,
        tokenizer=tokenize
    )


@pytest.fixture(scope="session")
def five38_tweets_tensors_cuda(five38_tweets: Five38TweetDataset, glove_embedding_cuda: WordEmbeddings):
    return Five38TweetTensorDataset(
        data_source=five38_tweets,
        embeddings=glove_embedding_cuda,
        tokenizer=tokenize
    )

###############################################################################


@pytest.fixture(scope="session")
def cresci_genuine_accounts_users():
    """Load genuine_accounts/users.csv from cresci-2017"""
    return CresciUserDataset(GENUINE_ACCOUNT_USER_PATH)


@pytest.fixture(scope="session")
def cresci_social_spambots_1_tweets():
    """Load social_spambots_1/tweets.csv from cresci-2017"""
    return CresciTweetDataset(SOCIAL_SPAMBOTS_1_TWEET_PATH)


@pytest.fixture(scope="session")
def cresci_social_spambots_1_users():
    """Load social_spambots_1/users.csv from cresci-2017"""
    return CresciUserDataset(SOCIAL_SPAMBOTS_1_USER_PATH)


@pytest.fixture(scope="session")
def cresci_genuine_accounts_tweets_tensors(request, device: torch.device):
    return request.getfixturevalue(f"cresci_genuine_accounts_tweets_tensors_{device.type}")


@pytest.fixture(scope="session")
def cresci_genuine_accounts_tweets_tensors_cpu(cresci_genuine_accounts_tweets, glove_embedding_cpu):
    return CresciTensorTweetDataset(
        data_source=cresci_genuine_accounts_tweets,
        embeddings=glove_embedding_cpu,
        tokenizer=tokenize
    )


@pytest.fixture(scope="session")
def cresci_genuine_accounts_tweets_tensors_cuda(cresci_genuine_accounts_tweets, glove_embedding_cuda):
    return CresciTensorTweetDataset(
        data_source=cresci_genuine_accounts_tweets,
        embeddings=glove_embedding_cuda,
        tokenizer=tokenize
    )


@pytest.fixture(scope="session")
def cresci_social_spambots_1_tweets_tensors(request, device: torch.device):
    return request.getfixturevalue(f"cresci_social_spambots_1_tweets_tensors_{device.type}")


@pytest.fixture(scope="session")
def cresci_social_spambots_1_tweets_tensors_cpu(cresci_social_spambots_1_tweets, glove_embedding_cpu):
    return CresciTensorTweetDataset(
        data_source=cresci_social_spambots_1_tweets,
        embeddings=glove_embedding_cpu,
        tokenizer=tokenize
    )


@pytest.fixture(scope="session")
def cresci_social_spambots_1_tweets_tensors_cuda(cresci_social_spambots_1_tweets, glove_embedding_cuda):
    return CresciTensorTweetDataset(
        data_source=cresci_social_spambots_1_tweets,
        embeddings=glove_embedding_cuda,
        tokenizer=tokenize
    )

###############################################################################
# training ####################################################################


@pytest.fixture(scope="session")
def make_trainer():
    def _make(device: torch.device, model: Module, optimizer: Optimizer):
        model.train(True)
        model.to(device)

        criterion = torch.nn.BCELoss()
        trainer = ignite.engine.create_supervised_trainer(model, optimizer, criterion, device)
        trainer.state = ignite.engine.State()

        @trainer.on(Events.STARTED)
        def set_model_in_state(trainer):
            trainer.state.model = model
            trainer.state.criterion = criterion

        @trainer.on(Events.COMPLETED)
        def finish_training(trainer):
            trainer.state.model.train(False)

        set_model_in_state(trainer)

        return trainer

    return _make


# @pytest.fixture(scope="function")
# def trainer(make_trainer, device: torch.device, lstm: ContextualLSTM):
#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=0.1,
#         momentum=0.9,
#         nesterov=True
#     )
#     return make_trainer(device, lstm, optimizer)

# ------------------------------------------------------------------------------
