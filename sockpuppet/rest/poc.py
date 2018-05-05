import requests
from flask import Blueprint, Flask, current_app
from flask_restful import Resource, reqparse, fields, inputs


def parse_ids(ids_arg):

    ids = tuple(map(inputs.natural, ids_arg.split(',')))

    return ids


parser = reqparse.RequestParser()
parser.add_argument(
    'ids',
    type=parse_ids,
    nullable=False,
    required=True,
    trim=True
)


blueprint = Blueprint("api", __name__)


# TODO: Cache this


class ProofOfConcept(Resource):
    def __init__(self):
        pass

    def get(self):
        args = parser.parse_args()
        current_app.logger.info(args)
        return {
            'hello': 'world',
            'args': args,
        }
