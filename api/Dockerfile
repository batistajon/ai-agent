FROM python:3.11-slim

RUN apt update && \
  apt install curl gcc -y

COPY requirements.txt /assistente/requirements.txt

WORKDIR /assistente

RUN python3 -m venv venv && \
  . venv/bin/activate

RUN pip install --upgrade pip && \
  pip install -r requirements.txt

ENV PYTHONPATH=/assistente

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]

