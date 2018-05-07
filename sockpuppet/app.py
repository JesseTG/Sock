# -*- coding: utf-8 -*-
"""The app module, containing the app factory function."""
from flask import Flask, render_template

from sockpuppet import commands, public, rest
from sockpuppet.extensions import api, cache, db, debug_toolbar, migrate, webpack
from sockpuppet.settings import ProdConfig
from botometer import Botometer


def create_app(config_object=ProdConfig):
    """An application factory, as explained here: http://flask.pocoo.org/docs/patterns/appfactories/.

    :param config_object: The configuration object to use.
    """
    app = Flask(__name__.split('.')[0])
    app.config.from_object(config_object)
    app.config.from_json(app.config['API_KEY_PATH'])
    register_extensions(app)
    register_blueprints(app)
    register_errorhandlers(app)
    register_shellcontext(app)
    register_commands(app)
    register_botometer(app)
    return app


def register_extensions(app):
    """Register Flask extensions."""
    api.init_app(rest.poc.blueprint)
    cache.init_app(app)
    db.init_app(app)
    debug_toolbar.init_app(app)
    migrate.init_app(app, db)
    webpack.init_app(app)

    # TODO: Can I do this in a more idiomatic way?
    api.add_resource(rest.poc.ProofOfConcept, "/get")

    return None


def register_blueprints(app):
    """Register Flask blueprints."""
    app.register_blueprint(public.views.blueprint)
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
            'db': db
        }

    app.shell_context_processor(shell_context)


def register_commands(app):
    """Register Click commands."""
    app.cli.add_command(commands.test)
    app.cli.add_command(commands.lint)
    app.cli.add_command(commands.clean)
    app.cli.add_command(commands.urls)


def register_botometer(app):
    app.botometer = Botometer(
        wait_on_ratelimit=True,
        mashape_key=app.config['MASHAPE_KEY'],
        consumer_key=app.config['TWITTER_CONSUMER_KEY'],
        consumer_secret=app.config['TWITTER_CONSUMER_SECRET'],
        access_token=app.config['TWITTER_ACCESS_TOKEN'],
        access_token_secret=app.config['TWITTER_ACCESS_TOKEN_SECRET']
    )
