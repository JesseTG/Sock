FROM python:3.7-slim

ARG MODEL_DEVICE
ENV SOCK_MODEL_DEVICE="${MODEL_DEVICE:-cpu}"

ARG SERVER_BIND_ADDRESS
ENV SOCK_SERVER_BIND_ADDRESS="${SERVER_BIND_ADDRESS:-ipc:///tmp/sock-server}"
# The address on which Sock accepts incoming requests (any ZMQ URL works)

ARG TRAINED_MODEL_PATH
ENV SOCK_TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH}"

ARG WORD_EMBEDDING_PATH
ENV SOCK_WORD_EMBEDDING_PATH="${WORD_EMBEDDING_PATH}"

# TODO: mount word embeddings and model as a volume

ARG WORKDIR
ENV WORKDIR="${WORKDIR:-/app}" 

HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 CMD [ "python3.7", "-m", "sock.cli.healthcheck" ]

WORKDIR ${WORKDIR}

COPY . ${WORKDIR}

# TODO: Open a UNIX socket for logging, and one for communication
RUN "${WORKDIR}/provision.sh"

CMD [ "python3.7", "-m", "main" ]