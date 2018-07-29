# -*- coding: utf-8 -*-
"""Defines fixtures available to all tests."""

import pytest
from webtest import TestApp

import torch
import ignite
from ignite.engine import Events, Engine

from sockpuppet.app import create_app
from sockpuppet.database import db as _db
from sockpuppet.settings import TestConfig
from sockpuppet.model.embedding import WordEmbeddings
from sockpuppet.model.nn.ContextualLSTM import ContextualLSTM
from sockpuppet.model.dataset.cresci import CresciTweetDataset, CresciUserDataset, CresciTensorTweetDataset
from sockpuppet.model.dataset.twitter_tokenize import tokenize

CRESCI_PATH = f"{TestConfig.TRAINING_DATA_PATH}/cresci-2017/datasets_full.csv"
GENUINE_ACCOUNT_TWEET_PATH = f"{CRESCI_PATH}/genuine_accounts.csv/tweets.csv"
GENUINE_ACCOUNT_USER_PATH = f"{CRESCI_PATH}/genuine_accounts.csv/users.csv"
SOCIAL_SPAMBOTS_1_TWEET_PATH = f"{CRESCI_PATH}/social_spambots_1.csv/tweets.csv"
SOCIAL_SPAMBOTS_1_USER_PATH = f"{CRESCI_PATH}/social_spambots_1.csv/users.csv"
GLOVE_PATH = f"{TestConfig.TRAINING_DATA_PATH}/glove/glove.twitter.27B.25d.txt"


def pytest_collection_modifyitems(config, items):
    if config.getoption("--profile"):
        # TODO: Detect the profiler plugin
        return
    skip_slow = pytest.mark.skip(reason="Only run if profiling")
    for item in items:
        if "profile" in item.keywords:
            item.add_marker(skip_slow)


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


@pytest.fixture(scope="session")
def glove_embedding():
    """Load the GloVe embeddings."""
    return WordEmbeddings(GLOVE_PATH, 25)


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
def cresci_genuine_accounts_tweets_tensors(cresci_genuine_accounts_tweets, glove_embedding):
    return CresciTensorTweetDataset(
        data_source=cresci_genuine_accounts_tweets,
        embeddings=glove_embedding,
        tokenizer=tokenize
    )


@pytest.fixture(scope="session")
def cresci_social_spambots_1_tweets_tensors(cresci_social_spambots_1_tweets, glove_embedding):
    return CresciTensorTweetDataset(
        data_source=cresci_social_spambots_1_tweets,
        embeddings=glove_embedding,
        tokenizer=tokenize
    )


@pytest.fixture(scope="session")
def cresci_genuine_accounts_tweets_tensors_cuda(cresci_genuine_accounts_tweets, glove_embedding):
    return CresciTensorTweetDataset(
        data_source=cresci_genuine_accounts_tweets,
        embeddings=glove_embedding,
        tokenizer=tokenize,
        device="cuda"
    )


@pytest.fixture(scope="session")
def cresci_social_spambots_1_tweets_tensors_cuda(cresci_social_spambots_1_tweets, glove_embedding):
    return CresciTensorTweetDataset(
        data_source=cresci_social_spambots_1_tweets,
        embeddings=glove_embedding,
        tokenizer=tokenize,
        device="cuda"
    )


@pytest.fixture(scope="function")
def trainer(glove_embedding: WordEmbeddings):
    def _trainer(device: str="cpu"):
        model = ContextualLSTM(glove_embedding)
        model.train(True)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01
        )
        criterion = torch.nn.CrossEntropyLoss()
        trainer = ignite.engine.create_supervised_trainer(model, optimizer, criterion, device)
        trainer.state = ignite.engine.State()

        @trainer.on(Events.STARTED)
        def set_model_in_state(trainer):
            trainer.state.model = model
            trainer.state.criterion = criterion

        set_model_in_state(trainer)

        return trainer

    return _trainer
