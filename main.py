# -*- coding: utf-8 -*-
"""Create an application instance."""


import asyncio
from enum import Enum
import logging
import signal
from threading import Event
import time
import torch
from torch import Tensor
from typing import Sequence


import zmq
from zmq import Frame, ZMQError
from zmq.log.handlers import PUBHandler
from zmq.asyncio import Context, Socket, Poller
from simplejson.errors import JSONDecodeError

from sockpuppet.model.data import WordEmbeddings, sentence_pad, PaddedSequence
from sockpuppet.model.nn import ContextualLSTM
from sockpuppet.model.serial import save, load
from sockpuppet.settings import DevConfig, ProdConfig

CONFIG = DevConfig if __debug__ else ProdConfig

# while true:
#  listen to requests
#  if request is valid:
#    guess each tweet, spit back response


class GuessRequest:
    def __init__(self, tweets: Sequence[str]):
        self.version = 0
        self.tweets = tweets


def deserialize(frames: Sequence[Frame]):
    return frames


def serialize() -> Sequence[Frame]:
    pass

async def main(embeddings: WordEmbeddings, model: ContextualLSTM, address: str, event: Event, context: Context):
    # TODO: Make logging configurable
    logging.info("Beginning main event loop")
    socket = context.socket(zmq.REP)  # type: Socket
    logging.info("Created %s", socket)
    socket.bind(address)
    logging.info("Bound socket to %s", address)

    while not event.is_set():
        # TODO: Don't busy loop, use poll or something
        # TODO: Event doesn't terminate loop if blocked in a socket call
        try:
            request = await socket.recv_json()
            logging.info("Received %s", request)
            if (not isinstance(request, list)) \
                    or len(request) == 0 \
                    or not all(isinstance(t, str) for t in request):
                await socket.send_json([])
                continue

            with torch.no_grad():
                # TODO: Put this in a task
                encoding = [embeddings.encode(e) for e in request]  # type: Sequence[Tensor]
                padded = sentence_pad(encoding)  # type: PaddedSequence
                guess_tensor = model(padded).round()  # type: Tensor
            guesses = tuple(bool(g) for g in guess_tensor)
            await socket.send_json(guesses)
            logging.info("Sent %s", guesses)
        except JSONDecodeError as e:
            logging.warning("Received invalid JSON")
            await socket.send_json([])
        except ZMQError as e:
            logging.warning("%s: %s", type(e).__name__, e)
        except Exception as e:
            logging.warning("%s: %s", type(e).__name__, e)
            break
        # TODO: Routing ID goes here once SERVER sockets are stable
        # TODO: Need a better protocol, mostly to handle errors

    socket.close()

# possible socket types for receving requests:
# - ZMQ_SERVER (most likely)
# - ZMQ_REP
# - pub
# - pull

if __name__ == '__main__':
    # TODO: Switch socket from ZMQ_REP to ZMQ_SERVER once it's stable
    # reason: ZMQ_SERVER scales better (responses don't have to immediately follow requests)

    context = Context.instance()  # type: Context
    log_handler = PUBHandler("tcp://*:5558", context=context)
    # TODO: May need to configure the underlying socket
    logging.basicConfig(
        format='[%(levelname)s %(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(), log_handler]
    )
    logging.info(f"zmq version {zmq.zmq_version()}")
    logging.info(f"pyzmq version {zmq.pyzmq_version()}")
    logging.info("Running Sock Puppet")

    embeddings = WordEmbeddings(CONFIG.WORD_EMBEDDING_PATH, CONFIG.MODEL_DEVICE)
    logging.info("Loaded %dD embeddings from %s onto %s", embeddings.dim,
                 CONFIG.WORD_EMBEDDING_PATH, embeddings.device)

    model = load(embeddings, CONFIG.TRAINED_MODEL_PATH, CONFIG.MODEL_DEVICE)
    event = Event()
    logging.info("Loaded trained model from %s onto %s", CONFIG.TRAINED_MODEL_PATH, model.device)
    asyncio.run(main(embeddings, model, CONFIG.SERVER_BIND_ADDRESS, event, context), debug=CONFIG.DEBUG)
