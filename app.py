from flask import Flask, render_template, Response, jsonify, request
import cv2

from safety_module import DrowsinessDetector
from routing_module import EVRouteOptimizer
from data_layer import fetch_and_enrich_routes

app = Flask(__name__)

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
            if not success:
                break
            processed_frame = detector.detect(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/evaluate_routes', methods=['POST'])
def evaluate_routes():
    """
    Score multiple OSRM routes.

    Expected JSON body:
    {
        "vehicle_name":     "Tata Nexon EV",
        "passenger_count":  2,
        "routes": [
            {
                "distance":      12.5,      // km
                "duration":      18.0,      // minutes
                "grade_percent":  2.5,      // optional, default 0
                "is_urban":       true      // optional, default false
            },
            ...
        ]
    }
    """
    data          = request.json
    routes_data   = data.get('routes', [])
    vehicle_name  = data.get('vehicle_name', 'Tata Nexon EV')

    try:
        passenger_count = int(data.get('passenger_count', 1))
    except ValueError:
        passenger_count = 1

    scores = optimizer.evaluate_alternatives(routes_data, vehicle_name, passenger_count)
    return jsonify({"scores": scores})


@app.route('/get_system_data')
def get_system_data():
    """
    Live dashboard polling endpoint.

    Query params:
        dist        — distance in km
        vehicle     — vehicle name
        passengers  — passenger count
        grade       — road grade percent (optional, default 0)
        urban       — '1' for urban driving (optional, default '0')
    """
    dist_val     = request.args.get('dist',       0)
    vehicle_name = request.args.get('vehicle',    'Tata Nexon EV')
    passengers   = request.args.get('passengers', 1)
    grade        = request.args.get('grade',      0.0)   # NEW
    urban        = request.args.get('urban',      '0')   # NEW

    status = detector.status

    try:
        p_count = int(passengers)
    except ValueError:
        p_count = 1

    try:
        grade_val = float(grade)
    except ValueError:
        grade_val = 0.0

    is_urban = (urban == '1')

    route_data = optimizer.get_optimal_route(
        status, dist_val, vehicle_name, p_count, grade_val, is_urban
    )

    return jsonify({
        "status":         status,
        "safety_warning": route_data.get("safety_warning"),  # None or warning dict
        "routes":         route_data
    })


@app.route('/nearest_station')
def nearest_station():
    """
    Returns the closest charging station to the driver's current location.
    Query params: lat, lon
    """
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing lat/lon"}), 400

    station = optimizer.get_nearest_station(lat, lon)
    return jsonify({"nearest_station": station})


@app.route('/get_routes', methods=['GET'])
def get_routes():
    """
    Step 3 — Full backend routing pipeline.

    The frontend only needs to send origin + destination.
    This endpoint does everything:
      1. Fetches up to 3 real routes from OSRM
      2. Gets elevation data for each route
      3. Scores each route using the physics energy model
      4. Ranks routes by energy (most efficient first)
      5. Attaches safety warning if driver is not Alert

    Query params:
        orig_lat, orig_lon  — origin coordinates
        dest_lat, dest_lon  — destination coordinates
        vehicle             — vehicle name (default: Tata Nexon EV)
        passengers          — passenger count (default: 1)
        urban               — '1' for urban driving (default: '0')
    """
    try:
        orig_lat = float(request.args.get('orig_lat'))
        orig_lon = float(request.args.get('orig_lon'))
        dest_lat = float(request.args.get('dest_lat'))
        dest_lon = float(request.args.get('dest_lon'))
    except (TypeError, ValueError):
        return jsonify({"error": "Missing or invalid coordinates"}), 400

    vehicle_name = request.args.get('vehicle', 'Tata Nexon EV')
    urban = request.args.get('urban','0') == '1'

    try:
        passenger_count = int(request.args.get('passengers', 1))
    except ValueError:
        passenger_count = 1

    # 1. Fetch routes from OSRM + enrich with elevation
    try:
        routes = fetch_and_enrich_routes(orig_lat, orig_lon, dest_lat, dest_lon)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    # 2. Score each route — use per-segment if available, else fallback
    for route in routes:
        if route.get("segments"):
            route["energy_kwh"] = optimizer.calculate_smart_energy_segmented(
                segments           = route["segments"],
                total_duration_min = route["duration_min"],
                total_distance_km  = route["distance_km"],
                vehicle_name       = vehicle_name,
                passenger_count    = passenger_count,
                is_urban           = urban
            )
        else:
            route["energy_kwh"] = optimizer.calculate_smart_energy(
                distance_km     = route["distance_km"],
                duration_min    = route["duration_min"],
                vehicle_name    = vehicle_name,
                passenger_count = passenger_count,
                grade_percent   = route.get("grade_percent", 0.0),
                is_urban        = urban
            )

    # 3. Rank by energy — most efficient first
    routes_ranked = sorted(routes, key=lambda r: r["energy_kwh"])

    # Mark the best route clearly
    if routes_ranked:
        routes_ranked[0]["recommended"] = True

    # 4. Attach safety warning if driver is impaired
    status        = detector.status
    DANGER_STATES = {"Drowsy", "Head Drop"}
    safety_warning = None

    if status in DANGER_STATES:
        safety_warning = {
            "alert":   True,
            "level":   "CRITICAL" if status == "Head Drop" else "WARNING",
            "reason":  status,
            "message": (
                "CRITICAL: Head drop detected — pull over immediately!"
                if status == "Head Drop"
                else "WARNING: Drowsiness detected — consider taking a break."
            )
        }

    return jsonify({
        "routes":         routes_ranked,
        "status":         status,
        "safety_warning": safety_warning,
        "vehicle":        vehicle_name,
        "passengers":     passenger_count
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True)