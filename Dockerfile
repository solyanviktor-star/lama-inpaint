FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip wget && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir runpod onnxruntime-gpu pillow numpy

# Download LaMa ONNX model
RUN mkdir -p /model && \
    wget -q -O /model/lama_fp32.onnx \
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
