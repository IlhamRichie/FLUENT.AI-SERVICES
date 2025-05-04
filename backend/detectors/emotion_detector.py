# detectors/emotion_detector.py

import cv2
import mediapipe as mp

def detect_emotion_status(image_path):
    image = cv2.imread(image_path)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not result.multi_face_landmarks:
            return "normal"

        landmarks = result.multi_face_landmarks[0].landmark
        
        # Ambil landmark untuk mulut bagian atas dan bawah
        upper_lip = landmarks[13]   # Upper lip
        lower_lip = landmarks[14]   # Lower lip
        
        # Hitung jarak vertikal antara bibir atas dan bawah
        lip_distance = abs(upper_lip.y - lower_lip.y)

        # Threshold jarak bibir: kalau terlalu besar -> mulut terbuka -> gugup
        if lip_distance > 0.04:  
            return "gugup"
        else:
            return "normal"
