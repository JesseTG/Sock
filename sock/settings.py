# -*- coding: utf-8 -*-
"""Application configuration."""
import os
import os.path


class Config(object):
    """Base configuration."""
    APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    DEBUG = False
    MODEL_DEVICE = os.environ.get("SOCK_MODEL_DEVICE", "cpu")
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))
    SERVER_BIND_ADDRESS = os.environ.get("SOCK_SERVER_BIND_ADDRESS", "ipc:///tmp/sock-server")
    TESTING = False
    TRAINED_MODEL_PATH = os.environ.get("SOCK_TRAINED_MODEL_PATH")
    TRAINING_DATA_PATH = os.environ.get("SOCK_TRAINING_DATA_PATH", os.path.expanduser("~/data"))
    WORD_EMBEDDING_PATH = os.environ.get("SOCK_WORD_EMBEDDING_PATH")


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
