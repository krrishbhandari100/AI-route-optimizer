import math

class EVRouteOptimizer:
    def __init__(self):
        # Emergency Charging Stations
        self.stations = [
            {"name": "Kelambakkam EV Station", "lat": 12.7916, "lon": 80.2201},
            {"name": "Sholinganallur Charging Hub", "lat": 12.9010, "lon": 80.2279}
        ]
        # Mathematical Constants
        self.BASE_KWH_PER_KM = 0.18  # Average EV consumption

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Haversine Formula: Calculates distance between any two global coordinates."""
        R = 6371 # Earth's radius
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def get_optimal_route(self, status, start_coords=None, end_coords=None):
        """
        Calculates Energy based on dynamic distance.
        Formula: Energy = Distance * (Base_Rate + Traffic_Penalty)
        """
        dist = 0
        if start_coords and end_coords:
            dist = self.calculate_distance(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
        
        # Simulated optimization: Comparing a standard vs an eco-tuned path
        energy_standard = dist * self.BASE_KWH_PER_KM * 1.2 # 20% more for high speed
        energy_eco = dist * self.BASE_KWH_PER_KM 

        return {
            "eco": {"energy": round(energy_eco, 2), "dist": round(dist, 1)},
            "fastest": {"energy": round(energy_standard, 2)},
            "suggested_stop": self.stations[1] if status in ["Drowsy", "Yawning"] else None
        }