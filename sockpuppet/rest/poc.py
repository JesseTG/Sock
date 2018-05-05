from flask import Blueprint, Flask, current_app
from flask_restful import Resource, reqparse, fields, inputs
from ..extensions import cache


def parse_ids(ids_arg):

    ids = frozenset(map(inputs.natural, ids_arg.split(',')))

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


def on_error(account, e):
    current_app.logger.error("ERROR")
    current_app.logger.error(account, e)


@cache.memoize()
def check_account(id):
    try:
        result = current_app.botometer.check_account(id)
        return {
            "id": str(id),
            "status": "ok",
            "likely": result["scores"]["english"]
        }
    except Exception as e:
        return {
            "id": str(id),
            "status": "error",
            "message": str(e),
        }


class ProofOfConcept(Resource):
    def __init__(self):
        pass

    def get(self):
        args = parser.parse_args()
        current_app.logger.info(args)
        guesses = tuple(check_account(id) for id in args.ids)
        current_app.logger.info(guesses)
        status = 200

        return {
            "results": guesses
        }, status
