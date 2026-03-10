from flask import Flask, render_template, jsonify, request
from routing_module import EVRouteOptimizer
from data_layer import fetch_and_enrich_routes

app = Flask(__name__)
optimizer = EVRouteOptimizer()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_vehicles')
def get_vehicles():
    """Returns all vehicle names loaded from ev_profiles.csv"""
    return jsonify({"vehicles": optimizer.get_vehicle_names()})


@app.route('/evaluate_routes', methods=['POST'])
def evaluate_routes():
    data            = request.json
    routes_data     = data.get('routes', [])
    vehicle_name    = data.get('vehicle_name', 'Tata Nexon EV')
    try:
        passenger_count = int(data.get('passenger_count', 1))
    except ValueError:
        passenger_count = 1
    scores = optimizer.evaluate_alternatives(routes_data, vehicle_name, passenger_count)
    return jsonify({"scores": scores})


@app.route('/get_routes', methods=['GET'])
def get_routes():
    """
    Full backend routing pipeline.
    1. Fetch up to 3 real routes from OSRM
    2. Enrich with elevation + per-segment grade
    3. Score each route using physics energy model
    4. Rank routes by energy (most efficient first)

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
    urban        = request.args.get('urban', '0') == '1'

    try:
        passenger_count = int(request.args.get('passengers', 1))
    except ValueError:
        passenger_count = 1

    # 1. Fetch routes from OSRM + enrich with elevation
    try:
        routes = fetch_and_enrich_routes(orig_lat, orig_lon, dest_lat, dest_lon)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    # 2. Score each route
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
    if routes_ranked:
        routes_ranked[0]["recommended"] = True

    return jsonify({
        "routes":     routes_ranked,
        "vehicle":    vehicle_name,
        "passengers": passenger_count
    })


if __name__ == '__main__':
    app.run(debug=True, threaded=True)