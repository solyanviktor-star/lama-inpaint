FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy==1.26.4 && \
    pip3 install --no-cache-dir runpod onnxruntime-gpu==1.17.1 pillow

RUN mkdir -p /model && \
    curl -L -o /model/lama_fp32.onnx \
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
