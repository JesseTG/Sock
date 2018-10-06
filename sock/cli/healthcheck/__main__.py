#!/usr/bin/env python3.7

import random
import sys

import lorem
import zmq
from zmq import Context, Socket

from sock.settings import ProdConfig


def main():
    context = Context.instance()  # type: Context
    socket = context.socket(zmq.REQ)  # type: Socket
    socket.connect(ProdConfig.SERVER_BIND_ADDRESS)

    num_tweets = random.randint(1, 30)
    tweets = tuple(lorem.sentence() for i in range(num_tweets))

    socket.send_json(tweets)
    guesses = socket.recv_json()

    if len(guesses) != num_tweets:
        raise RuntimeError(f"Expected to receive {num_tweets} guesses, got {len(guesses)}")

    # TODO: When I eventually store a hash of the embedding vectors in the
    # model, validate the hash here too

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)

    exit(0)
