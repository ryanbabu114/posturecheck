from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def check_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
    hip_slope = abs(left_hip.y - right_hip.y)

    correction_tips = []
    if shoulder_slope > 0.02:
        correction_tips.append("Keep shoulders level.")
    if hip_slope > 0.02:
        correction_tips.append("Align hips evenly.")

    if correction_tips:
        return "Wrong Posture", correction_tips
    else:
        return "Correct Posture", ["Great job!"]

@app.route('/posture-correction', methods=['POST'])
def posture_correction():
    try:
        file = request.files['frame']
        image_np = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Convert to RGB and process with Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            posture, corrections = check_posture(results.pose_landmarks.landmark)
            return jsonify({"status": "success", "posture": posture, "corrections": corrections})
        else:
            return jsonify({"status": "error", "message": "No posture detected"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
