#!/usr/bin/env python3.7

import random
import sys

import lorem
import zmq
from zmq import Context, Socket

from sock.settings import ProdConfig
from sock.utils import MIN_JSON_INT, MAX_JSON_INT


def main():
    context = Context.instance()  # type: Context
    socket = context.socket(zmq.REQ)  # type: Socket
    socket.connect(ProdConfig.SERVER_BIND_ADDRESS)

    num_tweets = random.randint(1, 30)
    id_number = random.randint(MIN_JSON_INT, MAX_JSON_INT)
    tweets = tuple(lorem.sentence() for i in range(num_tweets))
    request = {
        "jsonrpc": "2.0",
        "id": id_number,
        "method": "guess",
        "params": tweets
    }

    socket.send_json(request)
    response = socket.recv_json()  # type: dict

    if 'jsonrpc' not in response or response['jsonrpc'] != "2.0":
        raise ValueError("Response was not valid JSON-RPC 2.0")
    elif id_number != response.get("id"):
        raise ValueError(f"Expected to receive ID of {id_number}, got {response.get('id')}")
    elif "result" not in response:
        raise ValueError("No 'result' field found in response")
    elif len(response['result']) != num_tweets:
        raise RuntimeError(f"Expected to receive {num_tweets} guesses, got {len(response['result'])}")

    # TODO: When I eventually store a hash of the embedding vectors in the
    # model, validate the hash here too

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)

    exit(0)
