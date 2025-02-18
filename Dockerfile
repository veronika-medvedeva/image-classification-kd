FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

COPY . .
