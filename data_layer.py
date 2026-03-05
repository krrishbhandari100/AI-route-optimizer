import requests
import math

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
OSRM_BASE      = "https://router.project-osrm.org"
ELEVATION_BASE = "https://api.open-elevation.com/api/v1/lookup"
MAX_ALTERNATIVES = 3


# -------------------------------------------------------------------
# OSRM
# -------------------------------------------------------------------

def fetch_routes(origin_lat, origin_lon, dest_lat, dest_lon):
    coords = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    url    = f"{OSRM_BASE}/route/v1/driving/{coords}"
    params = {
        "alternatives": MAX_ALTERNATIVES,
        "steps":        "true",
        "geometries":   "geojson",
        "overview":     "full"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"OSRM request failed: {e}")

    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError("OSRM returned no valid routes.")

    routes = []
    for i, r in enumerate(data["routes"]):
        coords_latlon = [(pt[1], pt[0]) for pt in r["geometry"]["coordinates"]]

        # Extract turn-by-turn steps from all legs
        steps = []
        for leg in r.get("legs", []):
            for step in leg.get("steps", []):
                maneuver = step.get("maneuver", {})
                instruction = step.get("name", "")
                maneuver_type = maneuver.get("type", "")
                maneuver_mod  = maneuver.get("modifier", "")
                location      = maneuver.get("location", [0, 0])  # [lon, lat]
                distance_m    = step.get("distance", 0)

                # Build human readable instruction
                if maneuver_type == "turn":
                    text = f"Turn {maneuver_mod} onto {instruction}" if instruction else f"Turn {maneuver_mod}"
                elif maneuver_type == "depart":
                    text = f"Head {maneuver_mod} on {instruction}" if instruction else "Depart"
                elif maneuver_type == "arrive":
                    text = "You have arrived at your destination"
                elif maneuver_type == "roundabout":
                    text = f"Enter roundabout, take exit {maneuver.get('exit', '')}"
                else:
                    text = f"{maneuver_type.capitalize()} {maneuver_mod} {instruction}".strip()

                steps.append({
                    "text":       text,
                    "lat":        location[1],
                    "lon":        location[0],
                    "distance_m": round(distance_m)
                })

        routes.append({
            "index":        i,
            "distance_km":  round(r["distance"] / 1000, 3),
            "duration_min": round(r["duration"] / 60, 2),
            "geometry":     coords_latlon,
            "summary":      f"Route {i + 1}",
            "steps":        steps
        })
    return routes


# -------------------------------------------------------------------
# Elevation
# -------------------------------------------------------------------

def fetch_elevations(geometry):
    """
    Fetch elevation for sampled points along the route.
    Returns (sampled_points, elevations) tuple.
    Samples max 50 points evenly to stay within API limits.
    """
    total   = len(geometry)
    step    = max(1, total // 50)
    sampled = geometry[::step]

    locations = [{"latitude": lat, "longitude": lon} for lat, lon in sampled]

    try:
        resp = requests.post(
            ELEVATION_BASE,
            json={"locations": locations},
            timeout=15
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return sampled, [r["elevation"] for r in results]
    except requests.RequestException:
        return sampled, [0.0] * len(sampled)


# -------------------------------------------------------------------
# Per-segment grade  [NEW]
# -------------------------------------------------------------------

def compute_segments(sampled_points, elevations):
    """
    For each consecutive pair of sampled points compute:
        - distance_km
        - grade_percent  (rise/run × 100, capped ±30%)

    Returns a list of segment dicts used by the energy model.
    """
    def haversine_km(lat1, lon1, lat2, lon2):
        R    = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a    = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) \
               * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    segments = []
    for i in range(len(sampled_points) - 1):
        lat1, lon1 = sampled_points[i]
        lat2, lon2 = sampled_points[i + 1]

        dist_km   = haversine_km(lat1, lon1, lat2, lon2)
        elev_gain = elevations[i + 1] - elevations[i]
        dist_m    = dist_km * 1000

        grade = (elev_gain / dist_m * 100) if dist_m > 0 else 0.0
        grade = max(-30.0, min(30.0, grade))  # cap for stability

        segments.append({
            "distance_km":   round(dist_km, 4),
            "grade_percent": round(grade, 2),
            "elev_start":    elevations[i],
            "elev_end":      elevations[i + 1]
        })

    return segments


# -------------------------------------------------------------------
# Combined helper
# -------------------------------------------------------------------

def fetch_and_enrich_routes(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    Fetches OSRM routes and enriches each with:
        - elevations    — sampled elevation values
        - segments      — per-segment distance + grade  [NEW]
        - grade_percent — average grade for display
    """
    routes = fetch_routes(origin_lat, origin_lon, dest_lat, dest_lon)

    for route in routes:
        sampled_pts, elevations = fetch_elevations(route["geometry"])
        segments  = compute_segments(sampled_pts, elevations)
        avg_grade = (
            sum(s["grade_percent"] for s in segments) / len(segments)
            if segments else 0.0
        )

        route["elevations"] = elevations
        route["segments"] = segments
        route["grade_percent"] = round(avg_grade, 2)

    return routes