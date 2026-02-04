import cv2
import mediapipe as mp
import numpy as np

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.status = "Alert"
        self.EAR_THRESHOLD = 0.23
        self.MAR_THRESHOLD = 0.5
        self.frame_counter = 0
        self.DROWSY_LIMIT = 15 

    def calculate_ear(self, landmarks, eye_pts, w, h):
        pts = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_pts]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C)

    def calculate_mar(self, landmarks, mouth_pts, w, h):
        pts = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in mouth_pts]
        return np.linalg.norm(np.array(pts[0]) - np.array(pts[1])) / 100

    def detect(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_face_mesh.process(rgb)
        if res.multi_face_landmarks:
            for face in res.multi_face_landmarks:
                ear = (self.calculate_ear(face.landmark, [33, 160, 158, 133, 153, 144], w, h) + 
                       self.calculate_ear(face.landmark, [362, 385, 387, 263, 373, 380], w, h)) / 2
                mar = self.calculate_mar(face.landmark, [13, 14], w, h)
                if ear < self.EAR_THRESHOLD:
                    self.frame_counter += 1
                    if self.frame_counter >= self.DROWSY_LIMIT: self.status = "Drowsy"
                elif mar > self.MAR_THRESHOLD: self.status = "Yawning"
                else:
                    self.frame_counter = 0
                    self.status = "Alert"
        return frame