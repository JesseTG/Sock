# -*- coding: utf-8 -*-
"""Create an application instance."""


import asyncio
import concurrent.futures
import logging
import os
import getpass
import platform
import pathlib
import socket
import stat
import time
from collections import namedtuple
from functools import partial
from threading import Event
from os import stat_result
from typing import Sequence

import simplejson
import torch
import zmq
from jsonrpc import Dispatcher, JSONRPCResponseManager
from jsonrpc.exceptions import JSONRPCDispatchException, JSONRPCInvalidParams
from jsonrpc.jsonrpc2 import JSONRPC20Request, JSONRPC20Response
from simplejson.errors import JSONDecodeError
from torch import Tensor
from zmq import Frame, ZMQError
from zmq.asyncio import Context, Socket
from zmq.log.handlers import PUBHandler

from sock.model.data import PaddedSequence, WordEmbeddings, sentence_pad
from sock.model.nn import ContextualLSTM
from sock.model.serial import load, save
from sock.settings import DevConfig, ProdConfig

CONFIG = DevConfig if __debug__ else ProdConfig

Guess = namedtuple("Guess", ["id", "type", "status"])

SOCKET_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
SOCKET_DIR_PERMISSIONS = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH

async def main(
    embeddings: WordEmbeddings,
    model: ContextualLSTM,
    address: str,
    event: Event,
    context: Context,
    dispatcher: Dispatcher=Dispatcher()
):
    socket = context.socket(getattr(zmq, CONFIG.SERVER_SOCKET_TYPE))  # type: Socket
    logging.info("Created %s", socket)
    # TODO: Log more details about the socket
    # TODO: Put the entire socket in a with statement
    socket.bind(address)
    logging.info("Bound socket to %s", address)

    if address.startswith("ipc://"):
        # If this is a Unix socket...
        # TODO: Handle errors
        #  - socket path exists and is not a socket
        #  - cannot change socket permissions
        #  - cannot create socket, directories don't exist
        socket_path = pathlib.Path(address.replace("ipc://", ""))
        logging.info("Server address %s is a Unix socket", socket_path)

        socket_path.chmod(SOCKET_PERMISSIONS)
        socket_stat = socket_path.stat()  # type: stat_result
        logging.info("Set permissions to %o", socket_stat.st_mode)
        # TODO: Make this part more robust
        # - Use GIDs and UIDs
        # - Use umasks
        for p in list(socket_path.parents)[:-1]:
            try:
                p.chmod(SOCKET_DIR_PERMISSIONS)
                logging.info("Set permissions of %s to %o", p, SOCKET_DIR_PERMISSIONS)
            except:
                logging.warning("Failed to set permissions of %s to %o", p, SOCKET_DIR_PERMISSIONS)

    loop = asyncio.get_running_loop()

    @dispatcher.add_method
    def guess(*params: Sequence[str]) -> Sequence[bool]:
        if (not isinstance(params, tuple)) or (not all(isinstance(p, str) for p in params)) or len(params) == 0:
            # If we didn't get a list... (can't be a Sequence, we don't want just strings)
            # ...or if that list isn't made of tweets...
            # ...or if that list is empty...

            raise JSONRPCDispatchException(
                JSONRPCInvalidParams.CODE,
                JSONRPCInvalidParams.MESSAGE,
                data="params must be a nonempty tuple of strings"
            )

        logging.info("REQ: %d tweets", len(params))
        with torch.no_grad():
            encoding = [embeddings.encode(e) for e in params]  # type: Sequence[Tensor]
            padded = sentence_pad(encoding)  # type: PaddedSequence
            guess_tensor = model(padded).round()  # type: Tensor

        guesses = tuple(bool(g) for g in guess_tensor)

        return guesses

    @dispatcher.add_method
    def ping():
        logging.info("PING")
        return "pong"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # TODO: Do I really need this executor?
        logging.info("Beginning main event loop")
        while not event.is_set():
            # TODO: Don't busy loop, use poll or something
            # TODO: Event doesn't terminate loop if blocked in a socket call
            try:
                request = await socket.recv_string()  # type: str
                logging.debug("REQ: %s", request)
                # TODO: Log ID of request
                dispatch = partial(JSONRPCResponseManager.handle, request, dispatcher)
                # response = await loop.run_in_executor(executor, dispatch)  # type: JSONRPC20Response
                response = dispatch()  # type: JSONRPC20Response

                await socket.send_json(response.data)
                logging.info("REP: %s", response._id)
                logging.debug("REP: %s", response)

            except ZMQError as e:
                logging.exception("%s: %s", type(e).__name__, e)
            except Exception as e:
                logging.exception("%s: %s", type(e).__name__, e)
                break
            # TODO: Routing ID goes here once SERVER sockets are stable

    socket.close()
    # TODO: Delete the socket once the app ends
    logging.info("Socket at %s closed", address)

# possible socket types for receving requests:
# - ZMQ_SERVER (most likely)
# - ZMQ_REP
# - pub
# - pull

ZMQ_CAPABILITIES = ("ipc", "pgm", "tipc", "norm", "curve", "gssapi", "draft")

if __name__ == '__main__':
    # TODO: Switch socket from ZMQ_REP to ZMQ_SERVER once it's stable
    # reason: ZMQ_SERVER scales better (responses don't have to immediately follow requests)
    # TODO: Make server, when interrupted, print more informative info
    context = Context.instance()  # type: Context
    logging.basicConfig(
        format=CONFIG.LOG_FORMAT,
        datefmt=CONFIG.LOG_DATE_FORMAT,
        level=CONFIG.LOG_LEVEL,
        handlers=[logging.StreamHandler()]
    )

    # TODO: cpuinfo
    logging.info("ZMQ:")
    logging.info("  zmq version: %s", zmq.zmq_version())
    logging.info("  pyzmq version: %s", zmq.pyzmq_version())
    logging.info("  zmq includes: %s", zmq.get_includes())
    logging.info("  zmq library dirs: %s", zmq.get_library_dirs())
    logging.info("  has: %s", [c for c in ZMQ_CAPABILITIES if zmq.has(c)])
    logging.info("PyTorch:")
    logging.info("  version: %s", torch.__version__)
    logging.info("  CUDA available: %s", torch.cuda.is_available())
    logging.info("  CUDA version: %s", torch.version.cuda)
    logging.info("  CUDNN available: %s", torch.backends.cudnn.is_available())
    logging.info("  Debug build: %s", torch.version.debug)
    logging.info("  Default dtype: %s", torch.get_default_dtype())
    logging.info("  MKL available: %s", torch.backends.mkl.is_available())
    logging.info("  OMP threads: %d", torch.get_num_threads())
    logging.info("socket:")
    logging.info("  fqdn: %s", socket.getfqdn())
    logging.info("  has_ipv6: %s", socket.has_ipv6)
    logging.info("  hostname: %s", socket.gethostname())
    logging.info("  interfaces: %s", [i[1] for i in socket.if_nameindex()])
    logging.info("App config:")
    logging.info("  Environment: %s", CONFIG.__name__)
    logging.info("  MODEL_DEVICE: %s", CONFIG.MODEL_DEVICE)
    logging.info("  SERVER_BIND_ADDRESS: %s", CONFIG.SERVER_BIND_ADDRESS)
    logging.info("  SERVER_SOCKET_TYPE: %s", CONFIG.SERVER_SOCKET_TYPE)
    logging.info("  TRAINED_MODEL_PATH: %s", CONFIG.TRAINED_MODEL_PATH)
    logging.info("  WORD_EMBEDDING_PATH: %s", CONFIG.WORD_EMBEDDING_PATH)
    logging.info("os:")
    logging.info("  ctermid: %s", os.ctermid())
    logging.info("  cwd: %s", os.getcwd())
    logging.info("  groups: %s", os.getgroups())
    logging.info("  pgid: %d", os.getpgid(0))
    logging.info("  pgrp: %d", os.getpgrp())
    logging.info("  pid: %d", os.getpid())
    logging.info("  ppid: %d", os.getppid())
    logging.info("  priority_process: %d", os.getpriority(os.PRIO_PROCESS, 0))
    logging.info("  priority_pgrp: %d", os.getpriority(os.PRIO_PGRP, 0))
    logging.info("  priority_user: %d", os.getpriority(os.PRIO_USER, 0))
    logging.info("  resuid: ruid=%d, euid=%d, suid=%d", *os.getresuid())
    logging.info("  resgid: rgid=%d, egid=%d, sgid=%d", *os.getresgid())
    logging.info("  sid: %d", os.getsid(0))
    logging.info("  supports_bytes_environ: %s", os.supports_bytes_environ)
    logging.info("  uname: %s", os.uname())
    logging.info("  cpu_count: %d", os.cpu_count())
    logging.info("platform:")
    logging.info("  %s", platform.platform())
    logging.info("  python_build: %s", platform.python_build())
    logging.info("  python_compiler: %s", platform.python_compiler())
    logging.info("  python_branch: %s", platform.python_branch())
    logging.info("  python_implementation: %s", platform.python_implementation())
    logging.info("  python_revision: %s", platform.python_revision())
    logging.info("  python_version: %s", platform.python_version())
    logging.info("getpass:")
    logging.info("  user: %s", getpass.getuser())

    logging.info("Running Sock")

    embeddings = WordEmbeddings(CONFIG.WORD_EMBEDDING_PATH, CONFIG.MODEL_DEVICE)
    logging.info("Loaded %dD embeddings from %s onto %s", embeddings.dim,
                 CONFIG.WORD_EMBEDDING_PATH, embeddings.device)
    # TODO: Log memory usage on disk, and in RAM

    model = load(embeddings, CONFIG.TRAINED_MODEL_PATH, CONFIG.MODEL_DEVICE)
    # TODO: Load the embeddings into shared memory

    event = Event()
    logging.info("Loaded trained model from %s onto %s", CONFIG.TRAINED_MODEL_PATH, model.device)
    asyncio.run(main(embeddings, model, CONFIG.SERVER_BIND_ADDRESS, event, context), debug=CONFIG.DEBUG)
