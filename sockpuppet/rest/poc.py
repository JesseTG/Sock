from flask_restful import Resource
from flask import Flask, Blueprint


blueprint = Blueprint("api", __name__)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}
