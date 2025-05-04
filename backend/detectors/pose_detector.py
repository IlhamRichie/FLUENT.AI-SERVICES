# detectors/pose_detector.py

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

def detect_pose_status(image_path):
    """Detect head pose (straight or tilted)"""
    image = cv2.imread(image_path)
    if image is None:
        return "unknown"
    
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return "unknown"
    
    landmarks = results.multi_face_landmarks[0]
    
    # Get key points
    nose_tip = landmarks.landmark[4]
    left_face = landmarks.landmark[454]
    right_face = landmarks.landmark[234]
    
    # Calculate horizontal differences
    left_diff = abs(nose_tip.x - left_face.x)
    right_diff = abs(nose_tip.x - right_face.x)
    
    # Determine tilt
    tilt_ratio = abs(left_diff - right_diff) / max(left_diff, right_diff)
    
    return "miring" if tilt_ratio > 0.15 else "lurus"