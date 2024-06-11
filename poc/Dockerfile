FROM python:3.8-slim

RUN apt update && \
    apt install curl python-dev -y

WORKDIR /app

# COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip

EXPOSE 8000
