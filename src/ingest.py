'''
This module is responsible for reading the video file and sending it to the inference thread.
'''

import cv2
import threading
import logging
from src.models.data_models import FramePayload

logger = logging.getLogger(__name__)

class VideoIngestor(threading.Thread):
    def __init__(self, video_path: str, output_queue):
        super().__init__()
        self.video_path = video_path
        self.output_queue = output_queue
        # daemon=True ensures this thread dies automatically if you crash/stop the main program
        self.daemon = True 

    def run(self):
        logger.info(f"Ingestor starting. Reading video: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file: {self.video_path}")
            # Send the poison pill so the rest of the pipeline knows to shut down safely
            self.output_queue.put(None)
            return

        frame_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = cv2.VideoCapture(self.video_path)
                ret, frame = cap.read()
                if not ret:
                    break 
            
            # Create the standardized payload
            payload = FramePayload(frame_id=frame_id, image=frame)
            
            self.output_queue.put(payload)
            frame_id += 1
            
        cap.release()
        
        # Send a "Poison Pill" to tell the Inference thread the video is over
        self.output_queue.put(None) 
        print("[INFO] Video reading complete.")