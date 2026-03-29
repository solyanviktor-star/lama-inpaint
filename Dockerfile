FROM runpod/base:0.6.2-cuda11.8.0

RUN pip install --no-cache-dir runpod onnxruntime-gpu pillow numpy

RUN mkdir -p /model && \
    wget -q -O /model/lama_fp32.onnx \
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
