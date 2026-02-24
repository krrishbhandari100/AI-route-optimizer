import math

class EVRouteOptimizer:
    def __init__(self):
        # Professional EV Physics Profiles
        self.ev_profiles = {
            "Tata Nexon EV": {
                "base_mass_kg": 1400,
                "drag_cd": 0.33,
                "p_aux": 0.01  # Auxiliary energy per minute
            },
            "Tesla Model 3": {
                "base_mass_kg": 1610,
                "drag_cd": 0.23, # Highly aerodynamic
                "p_aux": 0.012 
            },
            "Mahindra XUV400": {
                "base_mass_kg": 1550,
                "drag_cd": 0.35,
                "p_aux": 0.011
            }
        }
        
        self.stations = [
            {"name": "VIT Charging Point", "lat": 12.8406, "lon": 80.1534},
            {"name": "Rest Area A1", "lat": 12.9500, "lon": 80.2000}
        ]

    def calculate_smart_energy(self, distance_km, duration_min, vehicle_name, passenger_count):
        """Advanced Physics-based energy consumption model"""
        car = self.ev_profiles.get(vehicle_name, self.ev_profiles["Tata Nexon EV"])
        
        # 1. Total Mass
        total_mass = car["base_mass_kg"] + (passenger_count * 75)
        
        # 2. Average Speed (km/h) - ensuring no division by zero
        avg_speed = distance_km / (duration_min / 60) if duration_min > 0 else 0
        
        # --- PHYSICS LOGIC ---
        
        # A. Rolling Resistance (Heavier car = more energy per km)
        # Using Nexon's 1400kg as the baseline 1.0 factor
        mass_factor = total_mass / 1400.0
        rolling_energy = distance_km * (0.15 * mass_factor)
        
        # B. Aerodynamic Drag (Continuous Curve)
        # Drag scales quadratically with speed for energy per km.
        # We baseline the drag at 40 km/h.
        speed_factor = (avg_speed / 40.0) ** 2  
        
        # We multiply distance by Drag Coefficient (Cd) and speed factor
        drag_energy = distance_km * (car["drag_cd"] * 0.1 * speed_factor)
        
        # C. Auxiliary Power (Traffic/Idling)
        aux_energy = duration_min * car["p_aux"]
        
        # D. Final Calculation
        total_energy_score = rolling_energy + drag_energy + aux_energy
        return round(total_energy_score, 3)

    def evaluate_alternatives(self, routes_data, vehicle_name, passenger_count):
        """Called by the frontend to score multiple OSRM routes"""
        evaluated_scores = []
        for r in routes_data:
            score = self.calculate_smart_energy(r['distance'], r['duration'], vehicle_name, passenger_count)
            evaluated_scores.append(score)
        return evaluated_scores

    def get_optimal_route(self, status, dist_from_map, vehicle_name="Tata Nexon EV", passenger_count=1):
        """Called every second by the sync function for live dashboard updates"""
        try:
            dist = float(dist_from_map) 
        except (ValueError, TypeError):
            dist = 0.0
            
        # Estimate duration based on a flat 40km/h for the live dashboard text
        duration_min = (dist / 40.0) * 60 if dist > 0 else 0
            
        energy = self.calculate_smart_energy(dist, duration_min, vehicle_name, passenger_count)
        
        return {
            "eco": {"dist": dist, "energy": energy},
            "suggested_stop": None
        }