FROM python:3.7-slim

ENV WORKDIR /app 
WORKDIR ${WORKDIR}

COPY . ${WORKDIR}

RUN pip3 install --no-cache-dir --requirement requirements/prod.txt

EXPOSE 80 443
