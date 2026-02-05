import cv2
import mediapipe as mp
import numpy as np
import time
import math
import collections

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Thresholds from your new file
        self.EAR_THRESH = 0.22
        self.EAR_CONSEC_FRAMES = 30 
        self.MAR_THRESH = 0.6
        self.PITCH_DROP_THRESH = 20.0
        
        # Counters & State
        self.eye_counter = 0
        self.head_drop_counter = 0
        self.yawn_timestamps = collections.deque()
        self.last_yawn_time = 0
        self.status = "Alert"
        
        # 3D Model Points for Head Pose
        self.MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
        ], dtype=np.float64)

    def get_head_pose(self, landmarks, shape):
        h, w = shape[:2]
        # Map 2D landmarks for PnP
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),   # nose
            (landmarks[152].x * w, landmarks[152].y * h), # chin
            (landmarks[33].x * w, landmarks[33].y * h),  # left eye
            (landmarks[263].x * w, landmarks[263].y * h), # right eye
            (landmarks[78].x * w, landmarks[78].y * h),  # left mouth
            (landmarks[308].x * w, landmarks[308].y * h)  # right mouth
        ], dtype=np.float64)

        focal_length = w
        camera_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype="double")
        _, rv, _ = cv2.solvePnP(self.MODEL_POINTS, image_points, camera_matrix, np.zeros((4,1)))
        rmat, _ = cv2.Rodrigues(rv)
        pitch = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
        return pitch

    def calculate_ear(self, landmarks, eye_indices, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h1 = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (v1 + v2) / (2.0 * h1)

    def detect(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_face_mesh.process(rgb)
        new_status = "Alert"

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            
            # 1. EAR Logic (Eyes)
            ear = (self.calculate_ear(lm, [33, 160, 158, 133, 153, 144], w, h) + 
                   self.calculate_ear(lm, [263, 387, 385, 362, 380, 373], w, h)) / 2.0
            
            if ear < self.EAR_THRESH:
                self.eye_counter += 1
                if self.eye_counter >= self.EAR_CONSEC_FRAMES: new_status = "Drowsy"
            else: self.eye_counter = 0

            # 2. MAR Logic (Yawn)
            mar = np.linalg.norm(np.array([lm[13].x*w, lm[13].y*h]) - np.array([lm[14].x*w, lm[14].y*h])) / \
                  (np.linalg.norm(np.array([lm[78].x*w, lm[78].y*h]) - np.array([lm[308].x*w, lm[308].y*h])) + 1e-6)
            
            if mar > self.MAR_THRESH:
                if (time.time() - self.last_yawn_time) > 2.0:
                    self.yawn_timestamps.append(time.time())
                    self.last_yawn_time = time.time()
                new_status = "Yawning"

            # 3. Head Pose (Pitch)
            try:
                pitch = self.get_head_pose(lm, (h, w))
                if pitch > self.PITCH_DROP_THRESH:
                    self.head_drop_counter += 1
                    if self.head_drop_counter >= 10: new_status = "Head Drop"
                else: self.head_drop_counter = 0
            except: pass

        self.status = new_status
        return frame