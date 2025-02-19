import os
import cv2
import numpy as np
import requests
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms

app = Flask(__name__)

# Define the ONNX model file path in Railway persistent storage
onnx_model_path = "/data/deepfake_model.onnx"
# Your GitHub Release URL for the model
download_url = "https://github.com/vabs-2004/Deepfake/releases/download/model/deepfake_model.onnx"

def download_model():
    if not os.path.exists(onnx_model_path):
        print("Downloading ONNX model from GitHub Releases...")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(download_url, headers=headers, stream=True, allow_redirects=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
            with open(onnx_model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download complete!")
            
            file_size = os.path.getsize(onnx_model_path)
            print("Downloaded file size (bytes):", file_size)
            # Check if file size is suspiciously low (e.g., < 300 MB)
            if file_size < 300 * 1024 * 1024:
                raise RuntimeError("Downloaded model file size is too small, indicating an incomplete download.")
        else:
            print(f"Failed to download model, status code: {response.status_code}")
            raise RuntimeError("ONNX model download failed.")

# Attempt to download the model if it's not present
download_model()

# Define image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ONNX model using ONNX Runtime
try:
    ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    print("ONNX model loaded successfully!")
except Exception as e:
    print("Error loading ONNX model:", str(e))
    raise RuntimeError("ONNX model could not be loaded.")

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    real_count, manipulated_count = 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Use 30 if FPS not available
    sample_rate = int(fps)  # Process one frame per second
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % sample_rate == 0:  # Apply frame sampling
            frame = cv2.resize(frame, (224, 224))
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(image).unsqueeze(0).numpy()

            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: image_tensor})
            predicted = np.argmax(outputs[0])

            if predicted == 0:
                real_count += 1
            else:
                manipulated_count += 1

        frame_index += 1

    cap.release()
    return "Real" if real_count > manipulated_count else "Manipulated"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    temp_video_path = "/data/temp_video.mp4"
    file.save(temp_video_path)

    result = predict_video(temp_video_path)
    os.remove(temp_video_path)

    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
