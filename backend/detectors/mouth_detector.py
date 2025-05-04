import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Mouth landmark indices
MOUTH_LANDMARKS = {
    'top_lip': [13, 312],
    'bottom_lip': [14, 317],
    'mouth_corners': [61, 291]
}

def calculate_mouth_open_ratio(landmarks, image_shape):
    """Calculate normalized mouth open ratio"""
    # Get relevant landmarks
    top_lip = landmarks.landmark[MOUTH_LANDMARKS['top_lip'][0]]
    bottom_lip = landmarks.landmark[MOUTH_LANDMARKS['bottom_lip'][0]]
    
    # Convert to pixel coordinates
    top_y = top_lip.y * image_shape[0]
    bottom_y = bottom_lip.y * image_shape[0]
    
    # Calculate vertical distance
    distance = abs(top_y - bottom_y)
    
    # Normalize using eye distance
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]
    eye_distance = abs(left_eye.x - right_eye.x) * image_shape[1]
    
    return distance / eye_distance

def detect_mouth_status(image_path):
    """Detect if mouth is open (speaking) or closed"""
    image = cv2.imread(image_path)
    if image is None:
        return "diam"
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return "diam"
    
    landmarks = results.multi_face_landmarks[0]
    ratio = calculate_mouth_open_ratio(landmarks, image.shape)
    
    # Dynamic threshold based on mouth corners
    mouth_corner_left = landmarks.landmark[MOUTH_LANDMARKS['mouth_corners'][0]]
    mouth_corner_right = landmarks.landmark[MOUTH_LANDMARKS['mouth_corners'][1]]
    mouth_width = abs(mouth_corner_right.x - mouth_corner_left.x)
    
    # Adjusted threshold
    threshold = 0.05 + (mouth_width * 0.1)  # Range ~0.06-0.08
    
    return "bicara" if ratio > threshold else "diam"