FROM mcr.microsoft.com/devcontainers/python:0-3.10
COPY requirements.txt .
RUN pip install -r requirements.txt
# requirementsでまとめてinstallしようとするとエラーになるので分けてinstallする
RUN pip install Chroma chromadb==0.3.29
RUN pip install tiktoken --upgrade
WORKDIR /workspace

# MOB Dockerfile production用も作成する
