import requests
from flask import Blueprint, Flask
from flask_restful import Resource, reqparse

parser = reqparse.RequestParser()
parser.add_argument('rate', type=int, help='Rate to charge for this resource')


blueprint = Blueprint("api", __name__)


# TODO: Cache this


class ProofOfConcept(Resource):
    def __init__(self):
        pass

    def get(self):
        args = parser.parse_args()
        return {
            'hello': 'world',
            'args': str(args),
        }
