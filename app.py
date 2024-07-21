from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import boto3
import os

app = Flask(__name__)
CORS(app)

# S3 configuration
S3_BUCKET = 'plantdetectionmodel'
MODEL_KEY = 'planting_detection_model.h5'
LOCAL_MODEL_PATH = '/tmp/planting_detection_model.h5'

# Function to download the model from S3
def download_model_from_s3(bucket, key, download_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, download_path)

# Download the model from S3
download_model_from_s3(S3_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)

# Load the plant detection model
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)

# Function to preprocess video frames for model prediction
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (150, 150))
    processed_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    processed_frame = processed_frame.astype(np.float32) / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)
    return processed_frame

# Endpoint to receive and process the recorded video
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'message': 'No video file received!'}), 400

    video_file = request.files['video']

    # Save the video file to a temporary location
    _, temp_path = tempfile.mkstemp(suffix='.webm')
    video_file.save(temp_path)

    # Open the saved video file
    cap = cv2.VideoCapture(temp_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess each frame
        preprocessed_frame = preprocess_frame(frame)
        frames.append(preprocessed_frame)

    cap.release()
    os.remove(temp_path)

    # Convert frames to numpy array
    frames = np.concatenate(frames, axis=0)

    # Perform predictions using the loaded model
    predictions = model.predict(frames)

    # Example: Check if planting action is detected (dummy condition)
    planting_detected = any(pred >= 0.5 for pred in predictions)

    if planting_detected:
        return jsonify({'message': 'Planting action detected!'}), 200
    else:
        return jsonify({'message': 'No planting action detected.'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
