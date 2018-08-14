# -*- coding: utf-8 -*-
"""Application configuration."""
import os
import os.path


class Config(object):
    """Base configuration."""

    SECRET_KEY = os.environ.get('SOCKPUPPET_SECRET', 'secret-key')  # TODO: Change me
    APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))
    DEBUG_TB_ENABLED = False  # Disable Debug toolbar
    DEBUG_TB_INTERCEPT_REDIRECTS = False
    CACHE_TYPE = 'redis'  # Can be "memcached", "redis", etc.
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WEBPACK_MANIFEST_PATH = 'webpack/manifest.json'
    CACHE_DEFAULT_TIMEOUT = 3600 * 72  # 3 days, given in seconds
    TWITTER_CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
    TWITTER_CONSUMER_SECRET = os.environ.get("TWITTER_CONSUMER_SECRET")
    TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
    TWITTER_ACCESS_TOKEN_SECRET = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
    MASHAPE_KEY = os.environ.get("MASHAPE_KEY")
    API_KEY_PATH = os.environ.get("SOCKPUPPET_API_KEY_PATH")
    CACHE_REDIS_HOST = os.environ.get("SOCKPUPPET_REDIS_HOST", "redis")
    CACHE_REDIS_PORT = os.environ.get("SOCKPUPPET_REDIS_PORT", 6379)
    CACHE_REDIS_PASSWORD = os.environ.get("SOCKPUPPET_REDIS_PASSWORD")
    CACHE_REDIS_DB = os.environ.get("SOCKPUPPET_REDIS_DB", 0)
    TRAINING_DATA_PATH = os.environ.get("SOCKPUPPET_TRAINING_DATA_PATH", os.path.expanduser("~/data"))


class ProdConfig(Config):
    """Production configuration."""

    ENV = 'prod'
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = 'postgresql://localhost/example'  # TODO: Change me
    DEBUG_TB_ENABLED = False  # Disable Debug toolbar


class DevConfig(Config):
    """Development configuration."""

    ENV = 'dev'
    DEBUG = True
    DB_NAME = 'dev.db'
    # Put the db file in project root
    DB_PATH = os.path.join(Config.PROJECT_ROOT, DB_NAME)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///{0}'.format(DB_PATH)
    DEBUG_TB_ENABLED = True
    CACHE_TYPE = 'simple'  # Can be "memcached", "redis", etc.


class TestConfig(Config):
    """Test configuration."""

    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite://'
