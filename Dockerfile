FROM python:3.7-slim

ENV WORKDIR /app 
WORKDIR ${WORKDIR}

COPY . ${WORKDIR}

# TODO: mount word embeddings and model as a volume
# TODO: Open a UNIX socket for logging, and one for communication
RUN pip3.7 install --no-cache-dir --compile --requirement requirements/prod.txt

CMD [ "python3.7", "-m", "main" ]