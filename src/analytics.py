import threading
import yaml
import math
from collections import deque, defaultdict
from shapely.geometry import Point, Polygon
from src.models.data_models import FramePayload
import logging

logger = logging.getLogger(__name__)

class Analytics(threading.Thread):
    def __init__(self, config_path: str, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.daemon = True
        
        # Load Config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Create Shapely Polygons
        self.zones = {
            name: Polygon(data["coordinates"]) 
            for name, data in self.config["zones"].items()
        }
        self.thresholds = {
            name: data["threshold"] 
            for name, data in self.config["zones"].items()
        }
        
        # State Management
        # Keep the last 15 frames of counts for each zone to smooth the data
        self.count_history = defaultdict(lambda: deque(maxlen=15))
        # Keep the last 10 positions of each track ID to calculate flow direction
        self.track_history = defaultdict(lambda: deque(maxlen=10))

    def _get_direction(self, dx, dy):
            """Converts vector differences to 8-way compass directions."""
            # Threshold to ignore tiny camera jitter
            if abs(dx) < 5 and abs(dy) < 5: 
                return "Stationary"
                
            angle = math.degrees(math.atan2(dy, dx))
            
            # 8-way slices (45 degrees each)
            if -22.5 <= angle < 22.5:
                return "East"
            elif 22.5 <= angle < 67.5:
                return "South-East"
            elif 67.5 <= angle < 112.5:
                return "South"
            elif 112.5 <= angle < 157.5:
                return "South-West"
            elif angle >= 157.5 or angle < -157.5:
                # West crosses the 180 / -180 boundary
                return "West"
            elif -157.5 <= angle < -112.5:
                return "North-West"
            elif -112.5 <= angle < -67.5:
                return "North"
            elif -67.5 <= angle < -22.5:
                return "North-East"
                
            return "Unknown"

    def run(self):
        logger.info("Analytics engine online.")
        while True:
            payload = self.input_queue.get()
            if payload is None:
                self.output_queue.put(None)
                break

            raw_counts = {zone: 0 for zone in self.zones}
            zone_flow_vectors = defaultdict(list)

            # Process every tracked person
            for track in payload.tracks:
                centroid = Point(track["centroid"])
                t_id = track["id"]
                
                # Update track history for flow
                self.track_history[t_id].append(track["centroid"])

                # Check which zone they are in
                for zone_name, polygon in self.zones.items():
                    if polygon.contains(centroid):
                        raw_counts[zone_name] += 1
                        
                        # Calculate their individual flow if we have enough history
                        history = self.track_history[t_id]
                        if len(history) >= 5:
                            dx = history[-1][0] - history[0][0]
                            dy = history[-1][1] - history[0][1]
                            zone_flow_vectors[zone_name].append((dx, dy))

            # Apply temporal smoothing and threshold checks
            for zone_name in self.zones:
                self.count_history[zone_name].append(raw_counts[zone_name])
                # Average the last 15 frames
                smoothed_count = int(sum(self.count_history[zone_name]) / len(self.count_history[zone_name]))
                payload.zone_counts[zone_name] = smoothed_count

                # Trigger Alert
                if smoothed_count > self.thresholds[zone_name]:
                    payload.alerts.append(
                        f"ALERT: {zone_name} exceeded threshold ({smoothed_count}/{self.thresholds[zone_name]})"
                    )

                # Aggregate dominant flow direction for the zone
                if zone_flow_vectors[zone_name]:
                    avg_dx = sum(v[0] for v in zone_flow_vectors[zone_name]) / len(zone_flow_vectors[zone_name])
                    avg_dy = sum(v[1] for v in zone_flow_vectors[zone_name]) / len(zone_flow_vectors[zone_name])
                    payload.zone_flows[zone_name] = self._get_direction(avg_dx, avg_dy)
                else:
                    payload.zone_flows[zone_name] = "None"

            self.output_queue.put(payload)