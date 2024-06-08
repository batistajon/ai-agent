FROM python:3.11-slim

RUN apt update && \
    apt install curl gcc -y

COPY requirements.txt requirements.txt
COPY app /app

RUN python3 -m venv venv && \
. venv/bin/activate

RUN pip install --upgrade pip && \
pip install -r requirements.txt

WORKDIR /app

ENV PYTHONPATH=/app

EXPOSE 5000

CMD uvicorn main:app --host 0.0.0.0 --port 5000 --reload
