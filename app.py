import os
import cv2
import numpy as np
import requests
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms

app = Flask(__name__)

# Define the ONNX model file name and download URL
onnx_model_path = "deepfake_model.onnx"
# Direct download URL from Google Drive
download_url = "https://drive.google.com/uc?id=1NJu6CkaOhsz_7xE4ztjdXJ_EFeQa_gCQ&export=download"

# Download the model if it doesn't exist
if not os.path.exists(onnx_model_path):
    print("Downloading ONNX model from Google Drive...")
    response = requests.get(download_url)
    if response.status_code == 200:
        with open(onnx_model_path, "wb") as f:
            f.write(response.content)
        print("Download complete!")
    else:
        print("Failed to download model, status code:", response.status_code)

# Define image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ONNX model using ONNX Runtime
ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    real_count, manipulated_count = 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Use 30 if FPS is 0 or not available
    sample_rate = int(fps)  # Process one frame per second
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % sample_rate == 0:  # Apply frame sampling
            # Resize frame before converting to reduce overhead
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
    temp_video_path = "temp_video.mp4"
    file.save(temp_video_path)

    result = predict_video(temp_video_path)
    os.remove(temp_video_path)

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
