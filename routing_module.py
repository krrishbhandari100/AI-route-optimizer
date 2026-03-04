import math

class EVRouteOptimizer:
    def __init__(self):
        self.ev_profiles = {
            "Tata Nexon EV": {
                "base_mass_kg": 1400,
                "drag_cd": 0.33,
                "frontal_area_m2": 2.45,
                "crr": 0.012,
                "regen_efficiency": 0.65,
                "drivetrain_efficiency": 0.85,
                "p_aux": 0.01
            },
            "Tesla Model 3": {
                "base_mass_kg": 1610,
                "drag_cd": 0.23,
                "frontal_area_m2": 2.22,
                "crr": 0.010,
                "regen_efficiency": 0.70,
                "drivetrain_efficiency": 0.90,
                "p_aux": 0.012
            },
            "Mahindra XUV400": {
                "base_mass_kg": 1550,
                "drag_cd": 0.35,
                "frontal_area_m2": 2.60,
                "crr": 0.013,
                "regen_efficiency": 0.60,
                "drivetrain_efficiency": 0.84,
                "p_aux": 0.011
            }
        }

        self.stations = [
            {"name": "VIT Charging Point", "lat": 12.8406, "lon": 80.1534},
            {"name": "Rest Area A1",        "lat": 12.9500, "lon": 80.2000}
        ]

        self.AIR_DENSITY = 1.225
        self.GRAVITY     = 9.81

    # ------------------------------------------------------------------
    # SINGLE SEGMENT energy (core physics)
    # ------------------------------------------------------------------

    def _segment_energy_joules(self, car, total_mass, dist_m, speed_ms, grade_percent, is_urban):
        """
        Computes energy in Joules for one road segment.
        Called repeatedly by calculate_smart_energy_segmented.
        """
        grade_rad = math.atan(grade_percent / 100.0)

        # A. Rolling Resistance
        F_rolling = car["crr"] * total_mass * self.GRAVITY * math.cos(grade_rad)
        E_rolling = F_rolling * dist_m

        # B. Aerodynamic Drag
        F_drag    = 0.5 * self.AIR_DENSITY * car["drag_cd"] * car["frontal_area_m2"] * (speed_ms ** 2)
        E_drag    = F_drag * dist_m

        # C. Gravity / Grade
        F_gravity = total_mass * self.GRAVITY * math.sin(grade_rad)
        E_gravity = F_gravity * dist_m
        if E_gravity < 0:
            E_gravity *= car["regen_efficiency"]  # downhill regen recovery

        # D. Kinetic stop-start losses (urban only)
        E_kinetic = 0.0
        if is_urban:
            dist_km       = dist_m / 1000
            num_stops     = dist_km * 1.0
            E_kinetic     = 0.5 * total_mass * (speed_ms ** 2) * num_stops
            E_kinetic    *= (1.0 - car["regen_efficiency"])

        E_total = E_rolling + E_drag + E_gravity + E_kinetic

        # Apply drivetrain efficiency
        if E_total > 0:
            E_total /= car["drivetrain_efficiency"]

        return E_total

    # ------------------------------------------------------------------
    # PER-SEGMENT scoring  [NEW]
    # ------------------------------------------------------------------

    def calculate_smart_energy_segmented(
        self,
        segments,           # list of {distance_km, grade_percent} dicts
        total_duration_min, # total route time (for speed estimate)
        total_distance_km,  # total route distance
        vehicle_name,
        passenger_count,
        is_urban=False
    ):
        """
        Scores energy by iterating over each segment individually.
        Each segment uses its own grade — far more accurate than
        using one average grade for the whole route.
        """
        car        = self.ev_profiles.get(vehicle_name, self.ev_profiles["Tata Nexon EV"])
        total_mass = car["base_mass_kg"] + (passenger_count * 75)

        # Estimate average speed from total distance/time
        avg_speed_ms = (
            (total_distance_km * 1000) / (total_duration_min * 60)
            if total_duration_min > 0 else 11.11
        )

        # Auxiliary energy for full route
        E_aux = (total_duration_min * 60) * (car["p_aux"] * 1000)

        # Sum energy across all segments
        E_segments = 0.0
        for seg in segments:
            dist_m = seg["distance_km"] * 1000
            if dist_m == 0:
                continue
            E_segments += self._segment_energy_joules(
                car, total_mass, dist_m,
                avg_speed_ms,
                seg["grade_percent"],
                is_urban
            )

        E_total_kWh = (E_segments + E_aux) / 3_600_000
        return round(E_total_kWh, 4)

    # ------------------------------------------------------------------
    # FALLBACK — whole-route scoring (used when segments unavailable)
    # ------------------------------------------------------------------

    def calculate_smart_energy(
        self,
        distance_km,
        duration_min,
        vehicle_name,
        passenger_count,
        grade_percent=0.0,
        is_urban=False
    ):
        car        = self.ev_profiles.get(vehicle_name, self.ev_profiles["Tata Nexon EV"])
        total_mass = car["base_mass_kg"] + (passenger_count * 75)
        distance_m = distance_km * 1000
        avg_speed_ms = (distance_m / (duration_min * 60)) if duration_min > 0 else 11.11

        E_main = self._segment_energy_joules(
            car, total_mass, distance_m, avg_speed_ms, grade_percent, is_urban
        )
        E_aux  = (duration_min * 60) * (car["p_aux"] * 1000)

        E_total_kWh = (E_main + E_aux) / 3_600_000
        return round(E_total_kWh, 4)

    # ------------------------------------------------------------------
    # PUBLIC API METHODS
    # ------------------------------------------------------------------

    def evaluate_alternatives(self, routes_data, vehicle_name, passenger_count):
        scores = []
        for r in routes_data:
            # Use segmented scoring if segments provided, else fallback
            if r.get("segments"):
                score = self.calculate_smart_energy_segmented(
                    segments           = r["segments"],
                    total_duration_min = r.get("duration", 0),
                    total_distance_km  = r.get("distance", 0),
                    vehicle_name       = vehicle_name,
                    passenger_count    = passenger_count,
                    is_urban           = r.get("is_urban", False)
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
        self,
        status,
        dist_from_map,
        vehicle_name    = "Tata Nexon EV",
        passenger_count = 1,
        grade_percent   = 0.0,
        is_urban        = False
    ):
        try:
            dist = float(dist_from_map)
        except (ValueError, TypeError):
            dist = 0.0

        duration_min = (dist / 40.0) * 60 if dist > 0 else 0

        # Live dashboard uses fallback (no segments available)
        energy = self.calculate_smart_energy(
            dist, duration_min, vehicle_name, passenger_count,
            grade_percent, is_urban
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