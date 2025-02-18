import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms

app = Flask(__name__)

# Define optimized image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert directly to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ONNX model
onnx_model_path = "deepfake_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    real_count, manipulated_count = 0, 0

    # Get FPS and set frame sampling rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:  # Handle cases where FPS metadata is missing
        fps = 30  # Assume a default FPS (adjust as needed)
    
    sample_rate = int(fps)  # Process one frame per second
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % sample_rate == 0:  # Apply frame sampling
            # Resize first to optimize processing
            frame = cv2.resize(frame, (224, 224))  
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(image).unsqueeze(0).numpy()

            # Run inference
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: image_tensor})
            predicted = np.argmax(outputs[0])

            # Count classification results
            if predicted == 0:
                real_count += 1
            else:
                manipulated_count += 1

        frame_index += 1

    cap.release()

    # Determine result based on majority of frames
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
