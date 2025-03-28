import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/')
def home():
    return "Posture Correction API is Live!"

@app.route('/posture-correction', methods=['POST'])
def posture_correction():
    try:
        file = request.files['frame']
        image_np = np.frombuffer(file.read(), np.uint8)
        frame = cv2.resize(frame, (320, 240))  # Reduce image size before processing


        # Convert to RGB and process with Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            return jsonify({"status": "success", "message": "Posture detected"})
        else:
            return jsonify({"status": "error", "message": "No posture detected"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get assigned Render port
    app.run(host='0.0.0.0', port=port, debug=True)
