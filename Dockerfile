FROM mcr.microsoft.com/devcontainers/python:0-3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download ja_core_news_lg
# MOB Dockerfile production用も作成する
WORKDIR /workspace