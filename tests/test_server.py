import asyncio
import logging
import time
from asyncio import Future
from importlib.resources import read_binary, read_text
from threading import Event, Thread
from typing import Sequence

import pytest
import zmq
from zmq.asyncio import Context, Socket
from zmq.log.handlers import PUBHandler
from zmq.utils.jsonapi import jsonmod

import main
from sock.model.data import WordEmbeddings
from sock.model.nn import ContextualLSTM
from sock.settings import TestConfig

DEFAULT_TIMEOUT = 3000  # ms


@pytest.fixture(scope="module")
def context():
    instance = Context.instance()  # type: Context
    log_handler = PUBHandler("tcp://*:5558", context=instance)
    logging.getLogger().addHandler(log_handler)

    return instance


@pytest.fixture(scope="function")
async def client(context: Context):
    socket = context.socket(zmq.REQ)
    socket.connect(TestConfig.SERVER_BIND_ADDRESS)

    yield socket

    socket.close()


@pytest.fixture(scope="function")
async def clients(context: Context):
    def make_socket():
        socket = context.socket(zmq.REQ)
        socket.connect(TestConfig.SERVER_BIND_ADDRESS)
        return socket

    sockets = tuple(make_socket() for i in range(10))

    yield sockets

    for s in sockets:
        s.close()


@pytest.fixture(scope="function")
def monitor(client: Socket):
    socket = client.get_monitor_socket()

    yield socket

    client.disable_monitor()


@pytest.fixture(scope="module")
def server(glove_embedding_cpu: WordEmbeddings, context: Context):
    event = Event()

    def _start_thread():
        lstm = ContextualLSTM(glove_embedding_cpu, device="cpu")
        asyncio.run(main.main(glove_embedding_cpu, lstm, TestConfig.SERVER_BIND_ADDRESS, event, context), debug=True)

    thread = Thread(target=_start_thread, daemon=True, name="Server")
    thread.start()

    yield thread

    event.set()


def test_server_is_running(server: Thread):
    assert server.is_alive()


@pytest.mark.usefixture("server")
def test_client_connects_to_server(client: Socket):
    assert client is not None


@pytest.mark.asyncio
@pytest.mark.usefixture("server")
async def test_no_unnecessary_blocking(client: Socket):
    request = jsonmod.loads(read_text("tests.data", "1-request.json"))

    start = time.time()
    assert (await client.send_json(request)) is None

    events = await client.poll(timeout=DEFAULT_TIMEOUT)
    assert events == 1

    response = await client.recv_json()
    duration = time.time() - start

    assert (duration * 1000) < DEFAULT_TIMEOUT


@pytest.mark.asyncio
@pytest.mark.usefixture("server")
@pytest.mark.parametrize("input", [
    "1-request.json",
    "2-request.json",
    "1-multiline.json",
    "2-multiline.json",
    "empty-tweets.json",
])
async def test_make_request(client: Socket, input: str):
    request = jsonmod.loads(read_text("tests.data", input))

    assert (await client.send_json(request)) is None

    events = await client.poll(timeout=DEFAULT_TIMEOUT)
    assert events == 1

    response = await client.recv_json()
    assert response is not None
    assert isinstance(response, Sequence)
    assert len(response) == len(request)
    assert all(isinstance(b, bool) for b in response)


@pytest.mark.asyncio
@pytest.mark.usefixture("server")
@pytest.mark.parametrize("input", [
    b'just some text',
    b'"quoted string"',
    b'\'single quoted string\'',
    b'',
    b'56',
    b'true',
    b'7.6',
    b'{}',
    b'[]',
    b'[',
    b'\0\0\0\0',
    b'{',
    b'null',
    read_binary("tests.data", "wrong-type.json"),
    pytest.param(b'\0' * 1000000, id="100000 null bytes"),
])
async def test_server_rejects_bad_request(client: Socket, input: bytes):
    assert (await client.send(input)) is None

    events = await client.poll(timeout=DEFAULT_TIMEOUT)
    assert events == 1

    response = await client.recv_json()
    assert response is not None
    assert isinstance(response, Sequence)
    assert len(response) == 0


@pytest.mark.asyncio
@pytest.mark.usefixture("server")
async def test_server_handles_concurrent_connections_sequential(clients: Sequence[Socket]):
    for i, socket in enumerate(clients, start=1):
        message = [str(i)] * i
        assert (await socket.send_json(message)) is None

        events = await socket.poll(timeout=DEFAULT_TIMEOUT)
        assert events == 1

        response = await socket.recv_json()
        assert response is not None
        assert isinstance(response, Sequence)
        assert len(response) == i


@pytest.mark.asyncio
@pytest.mark.usefixture("server")
async def test_server_handles_concurrent_connections_interleaved(clients: Sequence[Socket]):
    for i, socket in enumerate(clients, start=1):
        message = [str(i)] * i
        assert (await socket.send_json(message)) is None

    for i, socket in enumerate(clients, start=1):
        events = await socket.poll(timeout=DEFAULT_TIMEOUT)
        assert events == 1

    for i, socket in enumerate(clients, start=1):
        response = await socket.recv_json()
        assert response is not None
        assert isinstance(response, Sequence)
        assert len(response) == i


# @pytest.mark.asyncio
# async def test_server_handles_parallel_connections(server: Thread, context: Context):
#     pytest.xfail()
#     # TODO: Make N threads, spawn a socket on each thread and make a request each


def test_server_sends_responses_to_correct_clients(client: Socket):
    pytest.xfail()


@pytest.mark.asyncio
@pytest.mark.usefixture("server")
async def test_server_survives_dead_client(client: Socket, context: Context):
    request1 = jsonmod.loads(read_text("tests.data", "1-request.json"))
    request2 = jsonmod.loads(read_text("tests.data", "2-request.json"))

    dying_socket = context.socket(zmq.REQ)
    dying_socket.connect(TestConfig.SERVER_BIND_ADDRESS)

    assert (await dying_socket.send_json(request1)) is None

    dying_socket.close()

    assert (await client.send_json(request2)) is None

    events = await client.poll(timeout=DEFAULT_TIMEOUT)
    assert events == 1

    response = await client.recv_json()
    assert len(response) == len(request2)
