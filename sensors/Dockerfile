FROM python:3.9.5-slim-buster

ARG PACKAGE
ENV PACKAGE ${PACKAGE}
RUN mkdir /${PACKAGE}

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . /${PACKAGE}/
WORKDIR /${PACKAGE}

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

CMD python sender.py
