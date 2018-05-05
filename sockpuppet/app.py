# -*- coding: utf-8 -*-
"""The app module, containing the app factory function."""
from flask import Flask, render_template

from sockpuppet import commands, public, user, rest
from sockpuppet.twitter import twitter
from sockpuppet.extensions import api, bcrypt, cache, csrf_protect, db, debug_toolbar, login_manager, migrate, webpack
from sockpuppet.settings import ProdConfig


def create_app(config_object=ProdConfig):
    """An application factory, as explained here: http://flask.pocoo.org/docs/patterns/appfactories/.

    :param config_object: The configuration object to use.
    """
    app = Flask(__name__.split('.')[0])
    app.config.from_object(config_object)
    register_extensions(app)
    register_blueprints(app)
    register_errorhandlers(app)
    register_shellcontext(app)
    register_commands(app)
    init_twitter(app, config_object)
    return app


def register_extensions(app):
    """Register Flask extensions."""
    api.init_app(rest.poc.blueprint)
    bcrypt.init_app(app)
    cache.init_app(app)
    db.init_app(app)
    csrf_protect.init_app(app)
    login_manager.init_app(app)
    debug_toolbar.init_app(app)
    migrate.init_app(app, db)
    webpack.init_app(app)

    api.add_resource(rest.poc.ProofOfConcept, "/get")

    return None


def register_blueprints(app):
    """Register Flask blueprints."""
    app.register_blueprint(public.views.blueprint)
    app.register_blueprint(user.views.blueprint)
    app.register_blueprint(rest.poc.blueprint, url_prefix="/api/0")
    return None


def register_errorhandlers(app):
    """Register error handlers."""
    def render_error(error):
        """Render error template."""
        # If a HTTPException, pull the `code` attribute; default to 500
        error_code = getattr(error, 'code', 500)
        return render_template('{0}.html'.format(error_code)), error_code
    for errcode in [401, 404, 500]:
        app.errorhandler(errcode)(render_error)
    return None


def register_shellcontext(app):
    """Register shell context objects."""
    def shell_context():
        """Shell context objects."""
        return {
            'db': db,
            'User': user.models.User}

    app.shell_context_processor(shell_context)


def register_commands(app):
    """Register Click commands."""
    app.cli.add_command(commands.test)
    app.cli.add_command(commands.lint)
    app.cli.add_command(commands.clean)
    app.cli.add_command(commands.urls)


def init_twitter(app, config_object):
    twitter.SetCacheTimeout(config_object.TWITTER_CACHE_TIMEOUT)
    twitter.SetCredentials(
        config_object.TWITTER_CONSUMER_KEY,
        config_object.TWITTER_CONSUMER_SECRET,
        config_object.TWITTER_ACCESS_TOKEN,
        config_object.TWITTER_ACCESS_TOKEN_SECRET
    )

    app.logger.info(twitter.VerifyCredentials(False))
