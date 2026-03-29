import runpod
import torch
import numpy as np
import base64
import io
import time
from PIL import Image
import onnxruntime as ort

# Load model once at startup
print("Loading LaMa ONNX model...")
session = ort.InferenceSession(
    "/model/lama_fp32.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
print("Model loaded!")


def decode_image(data):
    """Decode base64 or data URI to PIL Image"""
    if data.startswith('data:'):
        data = data.split(',')[1]
    img_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')


def decode_mask(data):
    """Decode base64 or data URI to grayscale mask"""
    if data.startswith('data:'):
        data = data.split(',')[1]
    img_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(img_bytes)).convert('L')


def handler(job):
    start = time.time()
    inp = job['input']

    image = decode_image(inp['image'])
    mask = decode_mask(inp['mask'])

    orig_w, orig_h = image.size

    # Resize to 512x512 for model
    image_512 = image.resize((512, 512), Image.LANCZOS)
    mask_512 = mask.resize((512, 512), Image.NEAREST)

    # Convert to numpy
    img_np = np.array(image_512).astype(np.float32) / 255.0  # [H,W,3] -> 0-1
    mask_np = np.array(mask_512).astype(np.float32) / 255.0  # [H,W] -> 0-1

    # Reshape to [1, 3, 512, 512] and [1, 1, 512, 512]
    img_tensor = img_np.transpose(2, 0, 1)[np.newaxis, ...]  # [1,3,512,512]
    mask_tensor = mask_np[np.newaxis, np.newaxis, ...]  # [1,1,512,512]

    # Run inference
    results = session.run(None, {
        'image': img_tensor.astype(np.float32),
        'mask': mask_tensor.astype(np.float32)
    })

    output = results[0]  # [1, 3, 512, 512]

    # Detect output range
    max_val = np.max(np.abs(output[:, :, :100, :100]))
    if max_val > 2.0:
        output_uint8 = np.clip(output[0], 0, 255).astype(np.uint8)
    else:
        output_uint8 = np.clip(output[0] * 255, 0, 255).astype(np.uint8)

    # [3, 512, 512] -> [512, 512, 3]
    result_img = Image.fromarray(output_uint8.transpose(1, 2, 0))

    # Resize back to original
    if (orig_w, orig_h) != (512, 512):
        result_img = result_img.resize((orig_w, orig_h), Image.LANCZOS)

    # Encode to base64
    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    result_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    elapsed = round(time.time() - start, 2)
    print(f"Inpaint done in {elapsed}s")

    return {
        "image": result_b64,
        "time": elapsed
    }


runpod.serverless.start({"handler": handler})
