FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod onnxruntime pillow numpy

RUN mkdir -p /model && \
    curl -L -o /model/lama_fp32.onnx \
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
