# -*- coding: utf-8 -*-
"""Application configuration."""
import os
import os.path


class Config(object):
    """Base configuration."""
    APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    DEBUG = False
    MODEL_DEVICE = os.environ.get("SOCKPUPPET_MODEL_DEVICE", "cpu")
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))
    SERVER_BIND_ADDRESS = os.environ.get("SOCKPUPPET_SERVER_BIND_ADDRESS", "tcp://127.0.0.1:5555")
    TESTING = False
    TRAINED_MODEL_PATH = os.environ.get("SOCKPUPPET_TRAINED_MODEL_PATH")
    TRAINING_DATA_PATH = os.environ.get("SOCKPUPPET_TRAINING_DATA_PATH", os.path.expanduser("~/data"))
    WORD_EMBEDDING_PATH = os.environ.get("SOCKPUPPET_WORD_EMBEDDING_PATH")


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
    SERVER_BIND_ADDRESS = os.environ.get("SOCKPUPPET_SERVER_BIND_ADDRESS", "inproc://test-bind")
    TESTING = True
