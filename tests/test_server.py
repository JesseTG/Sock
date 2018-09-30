import pytest

from importlib.resources import read_text, read_binary
import time
from typing import Sequence
from threading import Event, Thread
import asyncio
from asyncio import Future

import zmq
from zmq.asyncio import Context, Socket
from zmq.utils.jsonapi import jsonmod

from sockpuppet.settings import TestConfig
from sockpuppet.model.data import WordEmbeddings
from sockpuppet.model.nn import ContextualLSTM

import main

DEFAULT_TIMEOUT = 3000  # ms


@pytest.fixture(scope="module")
def context():
    instance = Context.instance()  # type: Context
    return instance


@pytest.fixture(scope="function")
async def client(context: Context):
    socket = context.socket(zmq.REQ)
    socket.connect(TestConfig.SERVER_BIND_ADDRESS)

    yield socket

    socket.close()


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


def test_server_handles_multiple_clients_at_once(client: Socket):
    pytest.xfail()


def test_server_sends_responses_to_correct_clients(client: Socket):
    pytest.xfail()


def test_server_survives_dead_client(client: Socket):
    pytest.xfail()
