import cv2
import mediapipe as mp
import numpy as np
import math

class DrowsinessDetector:
    def __init__(self):
        # ── MediaPipe 0.10+ fix ──────────────────────────────────────
        # mp.solutions still works in 0.10 but needs explicit import
        self.mp_face_mesh_module = mp.solutions.face_mesh
        self.mp_face_mesh = self.mp_face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.EAR_THRESH        = 0.22
        self.EAR_CONSEC_FRAMES = 15
        self.MAR_THRESH        = 0.6
        self.PITCH_DROP_THRESH = 12.0
        self.PITCH_CONSEC_FRAMES = 3

        self.eye_closed_counter = 0
        self.head_drop_counter  = 0
        self.status = "Alert"

        self.MODEL_POINTS = np.array([
            (0.0,   0.0,   0.0),
            (0.0,  -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3,  32.7, -26.0),
            (-28.9,-28.9, -24.1),
            (28.9, -28.9, -24.1)
        ], dtype=np.float64)

    def euclid(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def get_head_pose(self, landmarks, h, w):
        image_points = np.array([
            (landmarks[1].x   * w, landmarks[1].y   * h),
            (landmarks[199].x * w, landmarks[199].y * h),
            (landmarks[33].x  * w, landmarks[33].y  * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[78].x  * w, landmarks[78].y  * h),
            (landmarks[308].x * w, landmarks[308].y * h)
        ], dtype=np.float64)

        camera_matrix = np.array(
            [[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double"
        )
        success, rv, _ = cv2.solvePnP(
            self.MODEL_POINTS, image_points,
            camera_matrix, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None
        rmat, _ = cv2.Rodrigues(rv)
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        return pitch

    def detect(self, frame):
        h, w, _ = frame.shape
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── 0.10+ requires passing image via context manager or direct call ─
        results = self.mp_face_mesh.process(rgb)
        current_status = "Alert"

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            def ear_calc(pts):
                v1  = self.euclid((lm[pts[1]].x, lm[pts[1]].y), (lm[pts[5]].x, lm[pts[5]].y))
                v2  = self.euclid((lm[pts[2]].x, lm[pts[2]].y), (lm[pts[4]].x, lm[pts[4]].y))
                hor = self.euclid((lm[pts[0]].x, lm[pts[0]].y), (lm[pts[3]].x, lm[pts[3]].y))
                return (v1 + v2) / (2.0 * hor + 1e-8)

            ear = (
                ear_calc([33,  160, 158, 133, 153, 144]) +
                ear_calc([263, 387, 385, 362, 380, 373])
            ) / 2.0

            if ear < self.EAR_THRESH:
                self.eye_closed_counter += 1
                if self.eye_closed_counter >= self.EAR_CONSEC_FRAMES:
                    current_status = "Drowsy"
            else:
                self.eye_closed_counter = 0

            mar = self.euclid((lm[13].x, lm[13].y), (lm[14].x, lm[14].y)) / \
                  (self.euclid((lm[78].x, lm[78].y), (lm[308].x, lm[308].y)) + 1e-8)
            if mar > self.MAR_THRESH:
                current_status = "Yawning"

            pitch = self.get_head_pose(lm, h, w)
            if pitch is not None and pitch > self.PITCH_DROP_THRESH:
                self.head_drop_counter += 1
                if self.head_drop_counter >= self.PITCH_CONSEC_FRAMES:
                    current_status = "Head Drop"
            else:
                self.head_drop_counter = 0

        self.status = current_status
        return frame