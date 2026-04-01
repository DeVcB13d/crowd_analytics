
import queue
import os
import argparse
import logging
from src.ingest import VideoIngestor
from src.inference import Inference
from src.analytics import Analytics
from src.renderer import Renderer


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Windows-specific fix for a common OpenMP error with some libraries


def main():
    parser = argparse.ArgumentParser(description="Crowd Analytics Pipeline")
    parser.add_argument("--config", type=str, default="configs/zones.yaml", help="Path to zones configuration")
    parser.add_argument("--model", type=str, default="weights/best.engine", help="Path to YOLO model engine")
    parser.add_argument("--video", type=str, default="samples/sample_video.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/final_annotated.mp4", help="Path to output annotated video")
    parser.add_argument("--log", type=str, default="logs/alerts.jsonl", help="Path to alerts log file")
    
    args = parser.parse_args()

    # File Paths
    CONFIG_PATH = args.config
    MODEL_PATH = args.model
    VIDEO_PATH = args.video
    OUTPUT_VIDEO_PATH = args.output
    LOG_PATH = args.log

    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

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