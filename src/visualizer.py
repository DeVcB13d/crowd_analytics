import cv2
import yaml
import numpy as np
from src.models.data_models import FramePayload

class Visualizer:
    def __init__(self, config_path: str):
        # Load Config for drawing zones
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.zones = {
            name: np.array(data["coordinates"], np.int32)
            for name, data in self.config["zones"].items()
        }
        self.thresholds = {
            name: data["threshold"] 
            for name, data in self.config["zones"].items()
        }

    def draw(self, payload: FramePayload):
        """Draws annotations (zones, tracks, counts) onto the frame."""
        frame = payload.image.copy()

        # 1. Draw Zones and Data
        for zone_name, pts in self.zones.items():
            count = payload.zone_counts.get(zone_name, 0)
            flow = payload.zone_flows.get(zone_name, "None")
            is_alerting = count > self.thresholds[zone_name]
            
            # Color: Red if alerting, Green if normal
            color = (0, 0, 255) if is_alerting else (0, 255, 0)
            
            # Draw Polygon
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            
            # Add Zone Overlay Text
            label = f"{zone_name}: {count} | Flow: {flow}"
            # Put text slightly above the first point of the polygon
            text_pos = (pts[0][0], pts[0][1] - 10) 
            cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 2. Draw Tracked People
        for track in payload.tracks:
            x1, y1, x2, y2 = track["box"]
            t_id = track["id"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(frame, f"ID:{t_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        return frame
