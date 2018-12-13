# -*- coding: utf-8 -*-
"""Application configuration."""
import os
import os.path
from pathlib import Path


def make_path(v):
    return Path(v) if v is not None else None

# TODO: Switch to environ module


class Config(object):
    """Base configuration."""
    APP_DIR = Path(os.path.abspath(os.path.dirname(__file__)))  # This directory
    DEBUG = False
    MODEL_DEVICE = os.environ.get("SOCK_MODEL_DEVICE", "cpu")
    LOG_FORMAT = os.environ.get("SOCK_LOG_FORMAT", '[%(levelname)s %(asctime)s] %(message)s')
    LOG_DATE_FORMAT = os.environ.get("SOCK_LOG_DATE_FORMAT", '%Y-%m-%d %H:%M:%S')
    LOG_LEVEL = os.environ.get("SOCK_LOG_LEVEL", "INFO")
    PROJECT_ROOT = Path(os.path.abspath(os.path.join(APP_DIR, os.pardir)))
    SERVER_BIND_ADDRESS = os.environ.get("SOCK_SERVER_BIND_ADDRESS", "ipc:///tmp/sock-server")
    SERVER_SOCKET_TYPE = os.environ.get("SOCK_SERVER_SOCKET_TYPE", "REP")
    TESTING = False
    TRAINED_MODEL_PATH = make_path(os.environ.get("SOCK_TRAINED_MODEL_PATH"))
    TRAINING_DATA_PATH = Path(os.environ.get("SOCK_TRAINING_DATA_PATH", os.path.expanduser("~/data")))
    WORD_EMBEDDING_PATH = make_path(os.environ.get("SOCK_WORD_EMBEDDING_PATH"))

    @classmethod
    def validate(cls):
        # TODO
        pass


class ProdConfig(Config):
    """Production configuration."""

    DEBUG = False
    ENV = 'prod'


class DevConfig(Config):
    """Development configuration."""

    DEBUG = True
    ENV = 'dev'


class TestConfig(Config):
    """Test configuration."""

    DEBUG = True
    SERVER_BIND_ADDRESS = os.environ.get("SOCK_SERVER_BIND_ADDRESS", "inproc://sock-server")
    TESTING = True
