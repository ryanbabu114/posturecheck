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

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def check_posture(landmarks, exercise):
    """Check posture based on the selected exercise type."""
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        correction_tips = []
        confidence = 100  # Start at full confidence

        if exercise == "Squat":
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            if knee_angle > 100:
                correction_tips.append("Bend knees more.")
            elif knee_angle < 60:
                correction_tips.append("Do not squat too low.")

        elif exercise == "Push-Up":
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            if elbow_angle > 160:
                correction_tips.append("Lower your body more.")
            elif elbow_angle < 90:
                correction_tips.append("Do not go too low.")

        elif exercise == "Plank":
            spine_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            if spine_angle < 170 or spine_angle > 190:
                correction_tips.append("Keep your back straight.")

        elif exercise == "Lunge":
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            if knee_angle > 110:
                correction_tips.append("Bend your knee more.")
            elif knee_angle < 70:
                correction_tips.append("Do not over-bend your knee.")

        if correction_tips:
            confidence -= len(correction_tips) * 10  # Reduce confidence by 10 per mistake
            return "Wrong Posture", correction_tips, confidence
        else:
            return "Correct Posture", ["Great job!"], confidence

    except Exception as e:
        return "Error", [str(e)], 0  # Return error message if something goes wrong

def process_frame(file, exercise):
    """Process frame for posture analysis."""
    image_np = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (640, 480))

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        posture, corrections, confidence = check_posture(results.pose_landmarks.landmark, exercise)
        return {"status": "success", "posture": posture, "corrections": corrections, "confidence": confidence}
    else:
        return {"status": "error", "message": "No posture detected"}

@app.route('/posture-correction', methods=['POST'])
def posture_correction():
    try:
        file = request.files['frame']
        exercise = request.form.get('exercise', 'Squat')  # Default to Squat
        result = process_frame(file, exercise)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
