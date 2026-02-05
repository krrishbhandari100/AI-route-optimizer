import math

class EVRouteOptimizer:
    def __init__(self):
        self.stations = [
            {"name": "VIT Charging Point", "lat": 12.8406, "lon": 80.1534},
            {"name": "Rest Area A1", "lat": 12.9500, "lon": 80.2000}
        ]

    # 'dist_from_map' parameter add kiya gaya
    def get_optimal_route(self, status, dist_from_map):
        try:
            # Placeholder 15.5 ko replace kiya
            dist = float(dist_from_map) 
        except (ValueError, TypeError):
            dist = 0.0
            
        # Energy Calculation: $Energy = Distance \times 0.15$
        energy = round(dist * 0.15, 2)
        
        return {
            "eco": {"dist": dist, "energy": energy},
            "suggested_stop": None
        }