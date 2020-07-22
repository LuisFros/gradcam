FROM python:3.7-slim-buster

RUN apt-get update && apt-get install -y python3-dev build-essential

RUN mkdir -p /code
WORKDIR /code

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000