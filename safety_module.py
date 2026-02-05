import cv2
import mediapipe as mp
import numpy as np
import math

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        
        # --- CONFIG REVERTED & TUNED FROM YOUR FILE ---
        self.EAR_THRESH = 0.22           #
        self.EAR_CONSEC_FRAMES = 15      # Faster than original 40 for responsiveness
        self.MAR_THRESH = 0.6            #
        self.PITCH_DROP_THRESH = 12.0    # Lowered from 20.0 for better sensitivity
        self.PITCH_CONSEC_FRAMES = 3     # Lowered from 10 for instant alert
        
        self.eye_closed_counter = 0
        self.head_drop_counter = 0
        self.status = "Alert"

        # Exact 3D Model Points from your script
        self.MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0),             # nose tip
            (0.0, -63.6, -12.5),         # chin
            (-43.3, 32.7, -26.0),        # left eye left corner
            (43.3, 32.7, -26.0),         # right eye right corner
            (-28.9, -28.9, -24.1),       # left mouth corner
            (28.9, -28.9, -24.1)         # right mouth corner
        ], dtype=np.float64)

    def euclid(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b)) #

    def get_head_pose(self, landmarks, h, w):
        # Precise mapping from your script
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),       # Nose
            (landmarks[199].x * w, landmarks[199].y * h),   # Chin
            (landmarks[33].x * w, landmarks[33].y * h),     # L Eye
            (landmarks[263].x * w, landmarks[263].y * h),   # R Eye
            (landmarks[78].x * w, landmarks[78].y * h),     # L Mouth
            (landmarks[308].x * w, landmarks[308].y * h)    # R Mouth
        ], dtype=np.float64)

        camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
        success, rv, _ = cv2.solvePnP(self.MODEL_POINTS, image_points, camera_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE) #
        
        if not success: return None
        rmat, _ = cv2.Rodrigues(rv) #
        pitch = math.degrees(math.atan2(rmat[2,1], rmat[2,2])) #
        return pitch

    def detect(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)
        current_status = "Alert"

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # EAR Calculation using your indices
            def ear_calc(pts):
                v1 = self.euclid((lm[pts[1]].x, lm[pts[1]].y), (lm[pts[5]].x, lm[pts[5]].y))
                v2 = self.euclid((lm[pts[2]].x, lm[pts[2]].y), (lm[pts[4]].x, lm[pts[4]].y))
                hor = self.euclid((lm[pts[0]].x, lm[pts[0]].y), (lm[pts[3]].x, lm[pts[3]].y))
                return (v1 + v2) / (2.0 * hor + 1e-8)

            ear = (ear_calc([33, 160, 158, 133, 153, 144]) + ear_calc([263, 387, 385, 362, 380, 373])) / 2.0 #

            if ear < self.EAR_THRESH:
                self.eye_closed_counter += 1
                if self.eye_closed_counter >= self.EAR_CONSEC_FRAMES: current_status = "Drowsy"
            else: self.eye_closed_counter = 0

            # MAR Calculation
            mar = self.euclid((lm[13].x, lm[13].y), (lm[14].x, lm[14].y)) / \
                  (self.euclid((lm[78].x, lm[78].y), (lm[308].x, lm[308].y)) + 1e-8)
            if mar > self.MAR_THRESH: current_status = "Yawning"

            # Improved Head Drop Logic
            pitch = self.get_head_pose(lm, h, w)
            if pitch is not None and pitch > self.PITCH_DROP_THRESH:
                self.head_drop_counter += 1
                if self.head_drop_counter >= self.PITCH_CONSEC_FRAMES:
                    current_status = "Head Drop"
            else:
                self.head_drop_counter = 0

        self.status = current_status
        return frame