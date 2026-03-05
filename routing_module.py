import math
import csv
import os

class EVRouteOptimizer:
    def __init__(self, profiles_csv=None):
        # ── Load EV profiles from CSV dataset ─────────────────────
        if profiles_csv is None:
            # Default: look for ev_profiles.csv next to this file
            profiles_csv = os.path.join(os.path.dirname(__file__), "ev_profiles.csv")

        self.ev_profiles = self._load_profiles(profiles_csv)

        # Fallback hardcoded profile if CSV fails to load
        self._fallback_profile = {
            "base_mass_kg": 1500,
            "drag_cd": 0.32,
            "frontal_area_m2": 2.45,
            "crr": 0.012,
            "regen_efficiency": 0.65,
            "drivetrain_efficiency": 0.85,
            "p_aux": 0.011
        }

        self.stations = [
            {"name": "VIT Charging Point", "lat": 12.8406, "lon": 80.1534},
            {"name": "Rest Area A1",        "lat": 12.9500, "lon": 80.2000}
        ]

        self.AIR_DENSITY = 1.225
        self.GRAVITY     = 9.81

    # ------------------------------------------------------------------
    # Load profiles from CSV
    # ------------------------------------------------------------------

    def _load_profiles(self, csv_path):
        """
        Reads ev_profiles.csv and returns a dict keyed by vehicle_name.
        Extra columns like cd_source, mass_source, crr_source are ignored
        by the physics model but kept in the file for academic citation.
        """
        profiles = {}
        PHYSICS_KEYS = {
            "base_mass_kg", "drag_cd", "frontal_area_m2",
            "crr", "regen_efficiency", "drivetrain_efficiency", "p_aux"
        }
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row["vehicle_name"].strip()
                    profiles[name] = {
                        k: float(row[k])
                        for k in PHYSICS_KEYS
                        if k in row and row[k].strip() != ''
                    }
            print(f"[EVRouteOptimizer] Loaded {len(profiles)} EV profiles from {csv_path}")
        except FileNotFoundError:
            print(f"[EVRouteOptimizer] WARNING: {csv_path} not found. Using fallback profile.")
        except Exception as e:
            print(f"[EVRouteOptimizer] WARNING: Failed to load profiles — {e}")
        return profiles

    def get_vehicle_names(self):
        """Returns list of all available vehicle names — used by frontend dropdown."""
        return sorted(self.ev_profiles.keys())

    # ------------------------------------------------------------------
    # Core physics — single segment
    # ------------------------------------------------------------------

    def _segment_energy_joules(self, car, total_mass, dist_m, speed_ms, grade_percent, is_urban):
        grade_rad = math.atan(grade_percent / 100.0)

        F_rolling = car["crr"] * total_mass * self.GRAVITY * math.cos(grade_rad)
        E_rolling = F_rolling * dist_m

        F_drag    = 0.5 * self.AIR_DENSITY * car["drag_cd"] * car["frontal_area_m2"] * (speed_ms ** 2)
        E_drag    = F_drag * dist_m

        F_gravity = total_mass * self.GRAVITY * math.sin(grade_rad)
        E_gravity = F_gravity * dist_m
        if E_gravity < 0:
            E_gravity *= car["regen_efficiency"]

        E_kinetic = 0.0
        if is_urban:
            num_stops  = (dist_m / 1000) * 1.0
            E_kinetic  = 0.5 * total_mass * (speed_ms ** 2) * num_stops
            E_kinetic *= (1.0 - car["regen_efficiency"])

        E_total = E_rolling + E_drag + E_gravity + E_kinetic
        if E_total > 0:
            E_total /= car["drivetrain_efficiency"]

        return E_total

    # ------------------------------------------------------------------
    # Per-segment energy scoring
    # ------------------------------------------------------------------

    def calculate_smart_energy_segmented(
        self, segments, total_duration_min, total_distance_km,
        vehicle_name, passenger_count, is_urban=False
    ):
        car        = self.ev_profiles.get(vehicle_name, self._fallback_profile)
        total_mass = car["base_mass_kg"] + (passenger_count * 75)
        avg_speed_ms = (
            (total_distance_km * 1000) / (total_duration_min * 60)
            if total_duration_min > 0 else 11.11
        )

        E_aux      = (total_duration_min * 60) * (car["p_aux"] * 1000)
        E_segments = 0.0

        for seg in segments:
            dist_m = seg["distance_km"] * 1000
            if dist_m == 0:
                continue
            E_segments += self._segment_energy_joules(
                car, total_mass, dist_m, avg_speed_ms,
                seg["grade_percent"], is_urban
            )

        return round((E_segments + E_aux) / 3_600_000, 4)

    # ------------------------------------------------------------------
    # Fallback whole-route scoring
    # ------------------------------------------------------------------

    def calculate_smart_energy(
        self, distance_km, duration_min, vehicle_name,
        passenger_count, grade_percent=0.0, is_urban=False
    ):
        car        = self.ev_profiles.get(vehicle_name, self._fallback_profile)
        total_mass = car["base_mass_kg"] + (passenger_count * 75)
        distance_m = distance_km * 1000
        avg_speed_ms = (distance_m / (duration_min * 60)) if duration_min > 0 else 11.11

        E_main = self._segment_energy_joules(
            car, total_mass, distance_m, avg_speed_ms, grade_percent, is_urban
        )
        E_aux  = (duration_min * 60) * (car["p_aux"] * 1000)

        return round((E_main + E_aux) / 3_600_000, 4)

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def evaluate_alternatives(self, routes_data, vehicle_name, passenger_count):
        scores = []
        for r in routes_data:
            if r.get("segments"):
                score = self.calculate_smart_energy_segmented(
                    r["segments"], r.get("duration", 0), r.get("distance", 0),
                    vehicle_name, passenger_count, r.get("is_urban", False)
                )
            else:
                score = self.calculate_smart_energy(
                    r.get("distance", 0), r.get("duration", 0),
                    vehicle_name, passenger_count,
                    r.get("grade_percent", 0.0), r.get("is_urban", False)
                )
            scores.append(score)
        return scores

    def get_optimal_route(
        self, status, dist_from_map,
        vehicle_name="Tata Nexon EV", passenger_count=1,
        grade_percent=0.0, is_urban=False
    ):
        try:
            dist = float(dist_from_map)
        except (ValueError, TypeError):
            dist = 0.0

        duration_min   = (dist / 40.0) * 60 if dist > 0 else 0
        energy         = self.calculate_smart_energy(
            dist, duration_min, vehicle_name, passenger_count, grade_percent, is_urban
        )

        DANGER_STATES  = {"Drowsy", "Head Drop"}
        safety_warning = None
        suggested_stop = None

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
            suggested_stop = self.stations[0]

        return {
            "eco":            {"dist": dist, "energy": energy},
            "suggested_stop": suggested_stop,
            "safety_status":  status,
            "safety_warning": safety_warning
        }

    def get_nearest_station(self, current_lat, current_lon):
        def haversine(lat1, lon1, lat2, lon2):
            R    = 6371
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a    = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) \
                   * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return min(
            self.stations,
            key=lambda s: haversine(current_lat, current_lon, s["lat"], s["lon"])
        )