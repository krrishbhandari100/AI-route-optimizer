import cv2
import mediapipe as mp
import numpy as np
import math

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        
        # --- FAST DETECTION THRESHOLDS ---
        self.EAR_THRESH = 0.25           # Thoda sensitive kiya
        self.EAR_CONSEC_FRAMES = 10      # 40 se 10 (Instant alarm)
        self.MAR_THRESH = 0.55           
        self.PITCH_THRESH = 20.0         # 20 degree drop from baseline
        self.PITCH_CONSEC_FRAMES = 5     # Fast head drop detection
        
        # Counters
        self.eye_counter = 0
        self.head_drop_counter = 0
        self.status = "Alert"
        
        # Calibration Variables
        self.baseline_pitch = None
        self.calib_frames = 0

    def euclid(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def get_head_pose(self, landmarks, h, w):
        # Indices from your uploaded file
        img_pts = np.array([
            (landmarks[1].x*w, landmarks[1].y*h),   # Nose
            (landmarks[152].x*w, landmarks[152].y*h), # Chin
            (landmarks[33].x*w, landmarks[33].y*h),  # L Eye
            (landmarks[263].x*w, landmarks[263].y*h), # R Eye
            (landmarks[78].x*w, landmarks[78].y*h),  # L Mouth
            (landmarks[308].x*w, landmarks[308].y*h)  # R Mouth
        ], dtype=np.float64)

        model_pts = np.array([(0.0,0.0,0.0), (0.0,-63.6,-12.5), (-43.3,32.7,-26.0),
                             (43.3,32.7,-26.0), (-28.9,-28.9,-24.1), (28.9,-28.9,-24.1)], dtype=np.float64)

        cam_mat = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
        _, rv, _ = cv2.solvePnP(model_pts, img_pts, cam_mat, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rv)
        return math.degrees(math.atan2(rmat[2,1], rmat[2,2]))

    def detect(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_face_mesh.process(rgb)
        current_status = "Alert"

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            
            # 1. EAR Logic (Optimized)
            def ear_calc(eye):
                p2, p6 = lm[eye[1]], lm[eye[5]]
                p3, p5 = lm[eye[2]], lm[eye[4]]
                p1, p4 = lm[eye[0]], lm[eye[3]]
                v = self.euclid((p2.x, p2.y), (p6.x, p6.y)) + self.euclid((p3.x, p3.y), (p5.x, p5.y))
                h_dist = self.euclid((p1.x, p1.y), (p4.x, p4.y))
                return v / (2.0 * h_dist + 1e-8)

            ear = (ear_calc([33,160,158,133,153,144]) + ear_calc([263,387,385,362,380,373])) / 2.0
            
            if ear < self.EAR_THRESH:
                self.eye_counter += 1
                if self.eye_counter >= self.EAR_CONSEC_FRAMES: current_status = "Drowsy"
            else: self.eye_counter = 0

            # 2. MAR Logic
            mar = self.euclid((lm[13].x, lm[13].y), (lm[14].x, lm[14].y)) / \
                  (self.euclid((lm[78].x, lm[78].y), (lm[308].x, lm[308].y)) + 1e-8)
            if mar > self.MAR_THRESH: current_status = "Yawning"

            # 3. Head Pitch with Calibration Fix
            try:
                pitch = self.get_head_pose(lm, h, w)
                if self.calib_frames < 40: # Pehle 40 frames calibration ke liye
                    if self.baseline_pitch is None: self.baseline_pitch = pitch
                    else: self.baseline_pitch = (self.baseline_pitch * 0.9) + (pitch * 0.1)
                    self.calib_frames += 1
                    current_status = "Calibrating..."
                else:
                    deviation = abs(pitch - self.baseline_pitch)
                    if deviation > self.PITCH_THRESH:
                        self.head_drop_counter += 1
                        if self.head_drop_counter >= self.PITCH_CONSEC_FRAMES: current_status = "Head Drop"
                    else: self.head_drop_counter = 0
            except: pass

        self.status = current_status
        return frame