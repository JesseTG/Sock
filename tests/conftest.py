# -*- coding: utf-8 -*-
"""Defines fixtures available to all tests."""

import csv
import sys
from typing import Callable, Sequence

import pytest
from pytest import Item, Session
from webtest import TestApp

import torch.multiprocessing
import torch
import ignite
from ignite.engine import Events, Engine

import pandas
from pandas import DataFrame

from sockpuppet.app import create_app
from sockpuppet.database import db as _db
from sockpuppet.settings import TestConfig
from sockpuppet.model.embedding import WordEmbeddings
from sockpuppet.model.nn.ContextualLSTM import ContextualLSTM
from sockpuppet.model.dataset.cresci import CresciTweetDataset, CresciUserDataset, CresciTensorTweetDataset
from sockpuppet.model.dataset.twitter_tokenize import tokenize
from .marks import needs_cuda, needs_cudnn

CRESCI_PATH = f"{TestConfig.TRAINING_DATA_PATH}/cresci-2017/datasets_full.csv"
GENUINE_ACCOUNT_TWEET_PATH = f"{CRESCI_PATH}/genuine_accounts.csv/tweets.csv"
GENUINE_ACCOUNT_USER_PATH = f"{CRESCI_PATH}/genuine_accounts.csv/users.csv"
SOCIAL_SPAMBOTS_1_TWEET_PATH = f"{CRESCI_PATH}/social_spambots_1.csv/tweets.csv"
SOCIAL_SPAMBOTS_1_USER_PATH = f"{CRESCI_PATH}/social_spambots_1.csv/users.csv"
GLOVE_PATH = f"{TestConfig.TRAINING_DATA_PATH}/glove/glove.twitter.27B.25d.txt"


def pytest_collection_modifyitems(config, items: Sequence[Item]):
    to_remove = set()
    for item in items:
        if "cpu_only" in item.keywords and item.callspec.params["device"] != "cpu":
            to_remove.add(item)
        elif "cuda_only" in item.keywords and not item.callspec.params["device"].startswith("cuda"):
            to_remove.add(item)

    for i in to_remove:
        items.remove(i)


def pytest_sessionstart(session: Session):
    if torch.cuda.is_available() and "spawn" in torch.multiprocessing.get_all_start_methods():
        torch.multiprocessing.set_start_method("spawn")


@pytest.fixture
def app():
    """An application for the tests."""
    _app = create_app(TestConfig)
    ctx = _app.test_request_context()
    ctx.push()

    yield _app

    ctx.pop()


@pytest.fixture
def testapp(app):
    """A Webtest app."""
    return TestApp(app)


@pytest.fixture
def db(app):
    """A database for the tests."""
    _db.app = app
    with app.app_context():
        _db.create_all()

    yield _db

    # Explicitly close DB connection
    _db.session.close()
    _db.drop_all()


@pytest.fixture
def user(db):
    """A user for the tests."""
    user = UserFactory(password='myprecious')
    db.session.commit()
    return user


@pytest.fixture(scope="session", params=["cpu", pytest.param("cuda", marks=[needs_cuda, needs_cudnn])])
def device(request):
    return request.param


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
def glove_embedding(device, glove_data: DataFrame):
    """Load the GloVe embeddings."""
    # TODO: Load the pandas frame from another source and pass that source into here
    return WordEmbeddings(glove_data, 25, device)


@pytest.fixture(scope="function")
def lstm(device: str, glove_embedding: WordEmbeddings):
    '''Creates a ContextualLSTM of either CPU or CUDA type'''
    lstm = ContextualLSTM(glove_embedding, device=device)
    return lstm


@pytest.fixture(scope="function")
def lstm_cuda(glove_embedding: WordEmbeddings):
    '''Creates a ContextualLSTM specifically on CUDA'''
    lstm = ContextualLSTM(glove_embedding, device="cuda")
    torch.cuda.synchronize()
    return lstm


@pytest.fixture(scope="function")
def lstm_cpu(glove_embedding: WordEmbeddings):
    lstm = ContextualLSTM(glove_embedding, device="cpu")
    return lstm


@pytest.fixture(scope="session")
def cresci_genuine_accounts_tweets():
    """Load genuine_accounts/tweets.csv from cresci-2017"""
    return CresciTweetDataset(GENUINE_ACCOUNT_TWEET_PATH)


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
def cresci_genuine_accounts_tweets_tensors(device, cresci_genuine_accounts_tweets, glove_embedding):
    return CresciTensorTweetDataset(
        data_source=cresci_genuine_accounts_tweets,
        embeddings=glove_embedding,
        tokenizer=tokenize,
        device=device
    )


@pytest.fixture(scope="session")
def cresci_social_spambots_1_tweets_tensors(device, cresci_social_spambots_1_tweets, glove_embedding):
    return CresciTensorTweetDataset(
        data_source=cresci_social_spambots_1_tweets,
        embeddings=glove_embedding,
        tokenizer=tokenize,
        device=device
    )


def _trainer(lstm: ContextualLSTM):
    lstm.train(True)
    optimizer = torch.optim.SGD(
        lstm.parameters(),
        lr=0.01
    )
    criterion = torch.nn.BCELoss()
    trainer = ignite.engine.create_supervised_trainer(lstm, optimizer, criterion, lstm.device)
    trainer.state = ignite.engine.State()

    @trainer.on(Events.STARTED)
    def set_model_in_state(trainer):
        trainer.state.model = lstm
        trainer.state.criterion = criterion

    set_model_in_state(trainer)

    return trainer


@pytest.fixture(scope="function")
def trainer(lstm: ContextualLSTM):
    return _trainer(lstm)


@pytest.fixture(scope="function")
def trainer_cpu(lstm_cpu: ContextualLSTM):
    return _trainer(lstm_cpu)


@pytest.fixture(scope="function")
def trainer_cuda(lstm_cuda: ContextualLSTM):
    return _trainer(lstm_cuda)
