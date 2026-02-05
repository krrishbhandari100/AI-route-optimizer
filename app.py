"""
AI EV Real-Time Navigator & Driver Safety System
Main Application Controller (Flask Backend)
"""

import cv2
import csv
import time
import os
from flask import Flask, render_template, Response, jsonify, request

# Importing our custom modules
from safety_module import DrowsinessDetector 
from routing_module import EVRouteOptimizer

app = Flask(__name__)

# --- MODULE INITIALIZATION ---
# detector: Handles Computer Vision (Mediapipe) for EAR, MAR, and Head Pitch
detector = DrowsinessDetector()

# optimizer: Handles Mathematical Route Optimization & Energy Calculation
optimizer = EVRouteOptimizer()

# --- LOGGING SETUP ---
# Creating a dedicated folder for forensic data persistence
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "safety_logs.csv")

# ----------------- VIDEO PROCESSING CORE -----------------

def generate_frames():
    """
    Continuous stream generator: Captures camera feed, processes it through AI,
    and yields JPEG frames to the frontend.
    """
    cap = cv2.VideoCapture(0) # 0 is the default webcam index
    
    # Initialize CSV header if file is new (Accountability Logic)
    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Driver_Status', 'Safety_Trigger'])

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # STEP 1: Process frame using Advanced Mediapipe Logic
            # This updates detector.status with "Drowsy", "Yawning", or "Head Drop"
            processed_frame = detector.detect(frame)
            
            # STEP 2: Data Logging (Forensic Persistence)
            # Log status every 5 seconds to monitor long-term driver behavior
            if int(time.time()) % 5 == 0:
                with open(LOG_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.ctime(), 
                        detector.status, 
                        "ACTIVE_REROUTE" if detector.status != "Alert" else "NONE"
                    ])

            # STEP 3: Encode for Web Streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Yielding multipart stream for <img> tag in HTML
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ----------------- API ENDPOINTS (The Logic Bridge) -----------------

@app.route('/')
def index():
    """Renders the main Dashboard UI."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for the real-time AI camera stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_system_data')
def get_system_data():
    """
    The Mathematical Bridge:
    1. Receives dynamic Source/Destination coordinates from the Frontend.
    2. Passes AI Status to the Optimizer to check if Rerouting is needed.
    3. Calculates Eco-Consumption (kWh) using the Haversine formula in routing_module.
    """
    # Extracting current map coordinates from URL parameters
    # Defaults set to VIT Chennai -> Marina Beach if no input provided
    slat = request.args.get('slat', default=12.8344, type=float)
    slon = request.args.get('slon', default=80.1530, type=float)
    elat = request.args.get('elat', default=13.0418, type=float)
    elon = request.args.get('elon', default=80.2841, type=float)

    # Triggering the Route Optimization Logic
    # status: Affects whether we suggest a Charging Station (Emergency Node)
    # coords: Used for real-time distance-to-energy calculation
    route_results = optimizer.get_optimal_route(
        status=detector.status, 
        start_coords=(slat, slon), 
        end_coords=(elat, elon)
    )

    # Return JSON response for seamless AJAX updates on the Dashboard
    return jsonify({
        "status": detector.status,
        "routes": route_results,
        "timestamp": time.time()
    })

if __name__ == '__main__':
    # Running the server on Localhost port 5000
    # Threading enabled to handle simultaneous video and data requests
    app.run(debug=True, port=5000, threaded=True)