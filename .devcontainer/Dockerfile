FROM python:3.10-slim

ENV PATH=$PATH:/root/.local/bin
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_VERSION=1.2.0

RUN apt-get update \
    && apt-get install -y curl gcc git tk-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python -
