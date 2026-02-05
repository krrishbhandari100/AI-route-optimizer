from flask import Flask, render_template, Response, jsonify, request
import cv2

# Import from your renamed file and class
from safety_module import DrowsinessDetector 
from routing_module import EVRouteOptimizer

app = Flask(__name__)

# Initialize Modules
detector = DrowsinessDetector()
optimizer = EVRouteOptimizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success: break
            # AI Processing
            processed_frame = detector.detect(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_system_data')
def get_system_data():
    # Frontend (index.html) se aane wala real distance
    dist_val = request.args.get('dist', 0) 
    
    status = detector.status # Drowsiness status
    
    # Optimizer ko map ka actual distance bheja
    route_data = optimizer.get_optimal_route(status, dist_val)
    
    return jsonify({"status": status, "routes": route_data})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)