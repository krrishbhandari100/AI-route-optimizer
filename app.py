import cv2
import csv
import time
from flask import Flask, render_template, Response, jsonify
from safety_module import DrowsinessDetector
from routing_module import EVRouteOptimizer

app = Flask(__name__)
detector = DrowsinessDetector()
optimizer = EVRouteOptimizer()

LOG_FILE = "safety_logs.csv"

def log_event(status):
    """Saves safety events to a CSV for the final project report."""
    if status in ["Drowsy", "Yawning"]:
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), status, "Rerouted to Station"])

# Initialize CSV Header
with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Driver_Status", "Action_Taken"])

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        frame = detector.detect(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_system_data')
def get_system_data():
    status = detector.status
    log_event(status)
    routes = optimizer.get_optimal_route(status)
    return jsonify({
        "status": status,
        "routes": routes
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)