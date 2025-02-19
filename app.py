import os
import cv2
import numpy as np
import requests
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms

app = Flask(__name__)

# ✅ Define the ONNX model file path in Railway persistent storage
onnx_model_path = "/data/deepfake_model.onnx"

# ✅ Your GitHub Release URL
download_url = "https://github.com/vabs-2004/Deepfake/releases/download/model/deepfake_model.onnx"

def download_model():
    if not os.path.exists(onnx_model_path):
        print("Downloading ONNX model from GitHub Releases...")
        headers = {"User-Agent": "Mozilla/5.0"}  # Prevents GitHub 403 errors
        response = requests.get(download_url, headers=headers, stream=True)

        if response.status_code == 200:
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
            with open(onnx_model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("✅ Download complete!")
        else:
            print(f"❌ Failed to download model, status code: {response.status_code}")
            raise RuntimeError("ONNX model download failed.")

# ✅ Download th
