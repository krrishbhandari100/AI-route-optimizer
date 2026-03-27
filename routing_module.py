import math
import csv
import os

class EVRouteOptimizer:
    def __init__(self, profiles_csv=None):
        if profiles_csv is None:
            profiles_csv = os.path.join(os.path.dirname(__file__), "ev_profiles.csv")

        self.ev_profiles = self._load_profiles(profiles_csv)

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
    # UNCERTAINTY LAYER
    # Sits on top of physics model — adds reserve, reliability metrics
    # ------------------------------------------------------------------

    def calculate_uncertainty_metrics(
        self,
        expected_kwh,
        distance_km,
        duration_min,
        grade_percent,
        is_urban,
        battery_kwh,
        soc_percent
    ):
        """
        Given expected kWh from physics model, compute:
        - confidence_reserve_kwh : extra buffer for congestion/terrain uncertainty
        - robust_kwh             : expected + reserve (worst case need)
        - available_kwh          : what battery has right now
        - arrival_soc_pct        : nominal arrival battery %
        - protected_arrival_pct  : arrival battery % after uncertainty reserve
        - trip_success_pct       : probability of completing trip safely

        Sources for uncertainty factors:
        - Base uncertainty: 8% of expected energy (standard EV range variance)
        - Urban stop-go adds 13% more uncertainty
        - Grade uncertainty: steep routes harder to predict
        - SAE J2452 / NREL RouteE variance estimates
        """

        available_kwh = battery_kwh * (soc_percent / 100.0)

        # ── Uncertainty factor build-up ───────────────────────────
        # Base: real-world EV energy varies ~8% from model predictions
        base_uncertainty = 0.08

        # Urban mode adds stop-start unpredictability
        urban_factor = 0.13 if is_urban else 0.0

        # Grade uncertainty — steeper roads harder to predict accurately
        grade_abs = abs(grade_percent)
        grade_factor = min(0.10, grade_abs * 0.008)

        # Distance factor — longer routes accumulate more uncertainty
        distance_factor = min(0.08, distance_km * 0.002)

        total_uncertainty = base_uncertainty + urban_factor + grade_factor + distance_factor

        # ── Confidence reserve ────────────────────────────────────
        confidence_reserve_kwh = round(expected_kwh * total_uncertainty, 4)

        # ── Robust kWh (worst case energy needed) ─────────────────
        robust_kwh = round(expected_kwh + confidence_reserve_kwh, 4)

        # ── Arrival SOC calculations ──────────────────────────────
        # Nominal: assumes best case (physics model exact)
        nominal_remaining_kwh   = available_kwh - expected_kwh
        arrival_soc_pct         = round((nominal_remaining_kwh / battery_kwh) * 100, 1)

        # Protected: assumes worst case (physics model + full reserve)
        protected_remaining_kwh = available_kwh - robust_kwh
        protected_arrival_pct   = round((protected_remaining_kwh / battery_kwh) * 100, 1)

        # ── Trip success probability ──────────────────────────────
        # Based on how much margin exists above the reserve need
        # If protected_arrival > 10% → high confidence
        # If protected_arrival 0-10% → medium confidence
        # If protected_arrival < 0% → trip is risky
        if protected_arrival_pct >= 15:
            trip_success_pct = round(min(98, 88 + (protected_arrival_pct - 15) * 0.5), 1)
        elif protected_arrival_pct >= 5:
            trip_success_pct = round(72 + (protected_arrival_pct - 5) * 1.6, 1)
        elif protected_arrival_pct >= 0:
            trip_success_pct = round(50 + protected_arrival_pct * 4.4, 1)
        else:
            # Protected arrival is negative — battery may not be enough
            trip_success_pct = round(max(8, 50 + protected_arrival_pct * 3.0), 1)

        # ── Risk level label ──────────────────────────────────────
        if trip_success_pct >= 85:
            risk_level = "Low Risk"
        elif trip_success_pct >= 70:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        return {
            "confidence_reserve_kwh": confidence_reserve_kwh,
            "robust_kwh":             robust_kwh,
            "available_kwh":          round(available_kwh, 3),
            "arrival_soc_pct":        arrival_soc_pct,
            "protected_arrival_pct":  protected_arrival_pct,
            "trip_success_pct":       trip_success_pct,
            "risk_level":             risk_level,
            "uncertainty_factor_pct": round(total_uncertainty * 100, 1)
        }

    # ------------------------------------------------------------------
    # Route flip threshold
    # At what battery % does the recommended route change?
    # ------------------------------------------------------------------

    def find_flip_threshold(self, routes, vehicle_name, passenger_count, is_urban, battery_kwh):
        """
        Scans SOC from 95% down to 5% to find the battery level
        where the recommended route changes (flip point).
        Returns flip_soc_pct and which routes swap.
        """
        def best_route_at_soc(soc):
            # Score each route: lower robust_kwh + better protected arrival = better
            best = None
            best_score = float('inf')
            for r in routes:
                u = self.calculate_uncertainty_metrics(
                    expected_kwh  = r["energy_kwh"],
                    distance_km   = r["distance_km"],
                    duration_min  = r["duration_min"],
                    grade_percent = r.get("grade_percent", 0.0),
                    is_urban      = is_urban,
                    battery_kwh   = battery_kwh,
                    soc_percent   = soc
                )
                # Score = robust need - protected arrival bonus
                score = u["robust_kwh"] - (u["protected_arrival_pct"] * 0.05)
                if score < best_score:
                    best_score = score
                    best = r
            return best

        high_soc_best = best_route_at_soc(95)
        flip_soc      = None
        flip_to       = None

        for soc in range(94, 4, -1):
            current_best = best_route_at_soc(soc)
            if current_best["index"] != high_soc_best["index"]:
                flip_soc = soc
                flip_to  = current_best
                break

        return {
            "has_flip":       flip_soc is not None,
            "flip_soc_pct":   flip_soc,
            "high_soc_route": high_soc_best.get("summary", "Route 1"),
            "low_soc_route":  flip_to.get("summary", "Route 2") if flip_to else None
        }

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