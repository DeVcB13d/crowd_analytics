
import queue
import os
import logging
from src.ingest import VideoIngestor
from src.inference import Inference
from src.analytics import Analytics
from src.renderer import Renderer


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Windows-specific fix for a common OpenMP error with some libraries


def main():
    # File Paths
    CONFIG_PATH = "configs/zones.yaml"
    MODEL_PATH = "weights/best.engine"
    VIDEO_PATH = "samples/sample_video.mp4"
    OUTPUT_VIDEO_PATH = "outputs/final_annotated.mp4"
    LOG_PATH = "logs/alerts.jsonl"

    # Ensure output directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Initializing Pipeline...")

    # 1. Create Thread-Safe Queues
    # maxsize acts as a shock absorber. If inference is slow, ingest pauses.
    frame_queue = queue.Queue(maxsize=30)
    detection_queue = queue.Queue(maxsize=30)
    output_queue = queue.Queue(maxsize=30)

    # 2. Instantiate Workers
    ingestor = VideoIngestor(
        video_path=VIDEO_PATH, 
        output_queue=frame_queue
    )
    
    inference = Inference(
        model_path=MODEL_PATH, 
        input_queue=frame_queue, 
        output_queue=detection_queue
    )
    
    analytics_engine = Analytics(
        config_path=CONFIG_PATH, 
        input_queue=detection_queue, 
        output_queue=output_queue
    )
    
    renderer = Renderer(
        config_path=CONFIG_PATH, 
        output_queue=output_queue, 
        output_video_path=OUTPUT_VIDEO_PATH, 
        log_path=LOG_PATH
    )

    # 3. Start the Background Threads
    ingestor.start()
    inference.start()
    analytics_engine.start()

    # 4. Run the Renderer on the Main Thread (Blocking)
    renderer.run()

    # 5. Graceful Shutdown
    ingestor.join()
    inference.join()
    analytics_engine.join()

if __name__ == "__main__":
    main()