import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,  # Increase threshold to ignore bad detections
    min_tracking_confidence=0.7
)


@app.route('/')
def home():
    return "Posture Correction API is Live!"

@app.route('/posture-correction', methods=['POST'])
def posture_correction():
    try:
        file = request.files['frame']
        image_np = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # ✅ Fix 1: Ensure Image Resolution is Large Enough
        frame = cv2.resize(frame, (320, 240))  # Increase resolution

        # ✅ Fix 2: Convert to RGB Before Processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # ✅ Fix 3: Debugging - Print Pose Landmarks
        if results.pose_landmarks:
            print("Pose landmarks detected:", results.pose_landmarks)
            return jsonify({"status": "success", "message": "Posture detected"})
        else:
            print("⚠ No Pose Detected!")
            return jsonify({"status": "error", "message": "No posture detected. Ensure full body is visible and lighting is good."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
