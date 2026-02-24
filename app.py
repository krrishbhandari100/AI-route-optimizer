from flask import Flask, render_template, Response, jsonify, request
import cv2

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

# NEW API: Evaluates multiple routes based on Vehicle Physics
@app.route('/evaluate_routes', methods=['POST'])
def evaluate_routes():
    data = request.json
    routes_data = data.get('routes', [])
    vehicle_name = data.get('vehicle_name', 'Tata Nexon EV')
    
    try:
        passenger_count = int(data.get('passenger_count', 1))
    except ValueError:
        passenger_count = 1
        
    # Send to the optimizer engine
    scores = optimizer.evaluate_alternatives(routes_data, vehicle_name, passenger_count)
    return jsonify({"scores": scores})

@app.route('/get_system_data')
def get_system_data():
    dist_val = request.args.get('dist', 0) 
    vehicle_name = request.args.get('vehicle', 'Tata Nexon EV')
    passengers = request.args.get('passengers', 1)
    
    status = detector.status # Drowsiness status
    
    try:
        p_count = int(passengers)
    except ValueError:
        p_count = 1
        
    route_data = optimizer.get_optimal_route(status, dist_val, vehicle_name, p_count)
    
    return jsonify({"status": status, "routes": route_data})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)