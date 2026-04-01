import cv2
import json
import yaml
import numpy as np
import time
from datetime import datetime
import os
import logging
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)

# Class to render the output of the pipeline
class Renderer:
    def __init__(self, config_path: str, output_queue, output_video_path: str, log_path: str):
        self.output_queue = output_queue
        self.output_video_path = output_video_path
        self.log_path = log_path
        self.video_writer = None
        self.visualizer = Visualizer(config_path)
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # Clear previous log file for a fresh run
        open(self.log_path, 'w').close() 

        # Performance Tracking
        self.prev_time = 0

    def _log_alerts(self, alerts, frame_id):
        """Writes structured JSON-lines to the log file."""
        if not alerts: return
        
        timestamp = datetime.now().isoformat()
        with open(self.log_path, 'a') as f:
            for alert in alerts:
                # Assuming alert string format from Analytics thread: 
                # "ALERT: ZoneName exceeded threshold (Count/Max)"
                zone_name = alert.split(" ")[1]
                log_entry = {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "zone_name": zone_name,
                    "event": "threshold_exceeded",
                    "message": alert
                }
                f.write(json.dumps(log_entry) + '\n')

    def run(self):
        logger.info("Renderer online. Press 'q' to stop.")
        
        while True:
            payload = self.output_queue.get()
            
            # Poison pill check
            if payload is None:
                break
                
            # Use the shared visualizer to annotate the frame
            frame = self.visualizer.draw(payload)

            # Performance Tracking
            curr_time = time.time()
            if self.prev_time > 0:
                fps = 1 / (curr_time - self.prev_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.prev_time = curr_time

            # Initialize VideoWriter once we know the frame size
            if self.video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, 30.0, (w, h))

            # Log Alerts
            self._log_alerts(payload.alerts, payload.frame_id)

            # Display and Save
            self.video_writer.write(frame)
            cv2.imshow("Crowd Analytics", frame)
            
            # Keep the UI responsive and listen for the quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        logger.info("Output saved and pipeline shutdown complete.")
