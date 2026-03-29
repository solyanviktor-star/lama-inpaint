import runpod
import numpy as np
import base64
import io
import time
import urllib.request
from PIL import Image

# Load model once at startup
print("Loading LaMa ONNX model...")
import onnxruntime as ort
session = ort.InferenceSession(
    "/model/lama_fp32.onnx",
    providers=['CPUExecutionProvider']
)
print(f"Model loaded! Providers: {session.get_providers()}")


def load_image(data):
    """Load image from base64 or URL"""
    if data.startswith('http'):
        req = urllib.request.Request(data, headers={'User-Agent': 'Mozilla/5.0'})
        img_bytes = urllib.request.urlopen(req).read()
        return Image.open(io.BytesIO(img_bytes)).convert('RGB')
    if data.startswith('data:'):
        data = data.split(',')[1]
    img_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')


def load_mask(data):
    """Load mask from base64 or URL"""
    if data.startswith('http'):
        req = urllib.request.Request(data, headers={'User-Agent': 'Mozilla/5.0'})
        img_bytes = urllib.request.urlopen(req).read()
        return Image.open(io.BytesIO(img_bytes)).convert('L')
    if data.startswith('data:'):
        data = data.split(',')[1]
    img_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(img_bytes)).convert('L')


def handler(job):
    try:
        start = time.time()
        inp = job['input']

        image = load_image(inp['image'])
        mask = load_mask(inp['mask'])

        orig_w, orig_h = image.size

        # Resize to 512x512
        image_512 = image.resize((512, 512), Image.LANCZOS)
        mask_512 = mask.resize((512, 512), Image.NEAREST)

        # To numpy [1, 3, 512, 512] float32 0-1
        img_np = np.array(image_512, dtype=np.float32) / 255.0
        img_tensor = img_np.transpose(2, 0, 1)[np.newaxis, ...]

        # Mask [1, 1, 512, 512] float32 0-1
        mask_np = np.array(mask_512, dtype=np.float32) / 255.0
        mask_tensor = mask_np[np.newaxis, np.newaxis, ...]

        # Inference
        t0 = time.time()
        results = session.run(None, {
            'image': img_tensor,
            'mask': mask_tensor
        })
        infer_time = round(time.time() - t0, 3)

        output = results[0]  # [1, 3, 512, 512]

        # Auto-detect range
        max_val = float(np.max(np.abs(output[0, :, :100, :100])))
        if max_val > 2.0:
            out_uint8 = np.clip(output[0], 0, 255).astype(np.uint8)
        else:
            out_uint8 = np.clip(output[0] * 255, 0, 255).astype(np.uint8)

        # [3, H, W] -> [H, W, 3]
        result_img = Image.fromarray(out_uint8.transpose(1, 2, 0))

        # Resize back
        if (orig_w, orig_h) != (512, 512):
            result_img = result_img.resize((orig_w, orig_h), Image.LANCZOS)

        # Encode
        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        result_b64 = base64.b64encode(buf.getvalue()).decode()

        total_time = round(time.time() - start, 2)
        print(f"Done: inference={infer_time}s, total={total_time}s")

        return {
            "image": result_b64,
            "inference_time": infer_time,
            "total_time": total_time
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
