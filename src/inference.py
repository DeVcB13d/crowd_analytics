import threading
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class Inference(threading.Thread):
    def __init__(self, model_path: str, input_queue, output_queue):
        super().__init__()
        # Load your ONNX engine
        self.model = YOLO(model_path, task='detect')
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.daemon = True

    def run(self):
        logger.info("Inferencer thread online.")
        while True:
            payload = self.input_queue.get()
            
            # Check for the poison pill
            if payload is None:
                self.output_queue.put(None)
                break

            # Run YOLO + ByteTrack
            # persist=True tells the tracker to remember IDs from the previous frame
            results = self.model.track(
                payload.image, 
                persist=True, 
                tracker="bytetrack.yaml", 
                verbose=False,
                device=0
            )

            # Extract tracking data if anyone was detected
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    # Calculate the bottom-center of the box (where their feet are)
                    centroid = (int((x1 + x2) / 2), int(y2))
                    
                    payload.tracks.append({
                        "id": track_id,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "centroid": centroid
                    })

            self.output_queue.put(payload)