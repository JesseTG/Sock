# -*- coding: utf-8 -*-
"""Defines fixtures available to all tests."""

import pytest
from webtest import TestApp

from sockpuppet.app import create_app
from sockpuppet.database import db as _db
from sockpuppet.settings import TestConfig

from .factories import UserFactory


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
