import os
import time
import queue
import threading
import cv2
import numpy as np
import logging
from typing import List, Dict, Any

# OPENMP Windows fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gradio as gr
from src.ingest import VideoIngestor
from src.inference import Inference
from src.analytics import Analytics
from src.models.data_models import FramePayload
from src.visualizer import Visualizer


# --- Global Pipeline Management ---
class PipelineManager:
    def __init__(self):
        self.threads = []
        self.stop_event = threading.Event()
        self.output_queue = queue.Queue(maxsize=30)
        self.visualizer = Visualizer("configs/zones.yaml")

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            if t.is_alive():
                # Note: Your worker classes should check for a stop signal
                # If they don't have one, they will close when the app exits
                pass 
        self.threads = []
        # Clear the queue to prevent old frames appearing on restart
        while not self.output_queue.empty():
            try: self.output_queue.get_nowait()
            except: break

manager = PipelineManager()

# --- Logic Functions ---

def start_pipeline(video_choice):
    """Initializes and kicks off the background workers."""
    manager.stop_event.clear()
    
    # Path Mapping
    video_path = "samples/pexels_videos_2740 (1080p).mp4" if "Sample" in video_choice else 0
    
    # Internal Queues
    frame_q = queue.Queue(maxsize=30)
    detect_q = queue.Queue(maxsize=30)
    
    # Instantiate Workers (Using your existing classes)
    ingestor = VideoIngestor(video_path=video_path, output_queue=frame_q)
    inference = Inference(model_path="weights/best.engine", input_queue=frame_q, output_queue=detect_q)
    analytics = Analytics(config_path="configs/zones.yaml", input_queue=detect_q, output_queue=manager.output_queue)

    # Start Threads
    manager.threads = [ingestor, inference, analytics]
    for t in manager.threads:
        t.daemon = True # Ensure threads exit if main process exits
        t.start()
    
    return "Pipeline Status: RUNNING"

def stream_vision():
    """Generator that yields data from the queue to the Gradio UI."""
    prev_time = time.time()
    last_alert = "System Nominal"
    last_frame = None
    last_metrics = "Waiting for detections..."

    try:
        while not manager.stop_event.is_set():
            try:
                # Wait for a payload with a short timeout to keep the loop responsive
                payload: FramePayload = manager.output_queue.get(timeout=0.1)
                
                # 1. Process Image & Annotate
                annotated_frame = manager.visualizer.draw(payload)
                
                # OPTIMIZATION: Downscale for the web to improve FPS
                # Streaming 1080p over a websocket is very slow. 720p or 540p is better.
                h, w = annotated_frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    annotated_frame = cv2.resize(annotated_frame, (1280, int(h * scale)))

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                last_frame = frame_rgb
                
                # 2. Performance Tracking
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                # 3. Format Zone Metrics & Flow
                stats_lines = []
                for zone, count in payload.zone_counts.items():
                    flow = payload.zone_flows.get(zone, "N/A")
                    stats_lines.append(f"📍 {zone.upper()}\n   Count: {count} | Flow: {flow}")
                
                last_metrics = "\n\n".join(stats_lines) if stats_lines else "No zones active"

                # 4. Handle Alerts
                if payload.alerts:
                    last_alert = f"⚠️ ALERT: " + " | ".join(payload.alerts)
                
                yield (
                    frame_rgb,
                    last_metrics,
                    last_alert,
                    f"Throughput: {fps:.1f} FPS"
                )
                
                # OPTIONAL: Reduced Throttling
                time.sleep(0.01) 

            except queue.Empty:
                # If the queue is empty, just yield the last state to keep the UI alive
                if last_frame is not None:
                    yield (
                        last_frame,
                        last_metrics,
                        last_alert,
                        "Throughput: 0 FPS (Waiting...)"
                    )
                continue
            except Exception as e:
                print(f"Streaming Error: {e}")
                time.sleep(0.1)
                continue
    finally:
        # If the browser closes, this code runs automatically. 
        # We signal all background threads to stop immediately.
        print("Gradio client disconnected. Shutting down pipeline.")
        manager.stop()

def stop_pipeline():
    manager.stop()
    return "Pipeline Status: STOPPED", None, "System Cleared", "Throughput: 0 FPS"

# --- Gradio UI Definition ---

with gr.Blocks(theme=gr.themes.Default(primary_hue="orange"), title="VariPhi AI Console") as demo:
    gr.Markdown("# Crowd Analytics")
    
    with gr.Row():
        # Left Side: Video and Alerts
        with gr.Column(scale=3):
            video_output = gr.Image(label="Annotated Live Feed")
            alert_banner = gr.Label(value="System Initialized", label="Security Events")
            
        # Right Side: Metrics and Controls
        with gr.Column(scale=1):
            status_label = gr.Markdown("Pipeline Status: IDLE")
            fps_box = gr.Markdown("Throughput: 0 FPS")
            
            stream_input = gr.Dropdown(
                choices=["Sample Video", "Webcam / Live Stream"], 
                label="Select Input Source", 
                value="Sample Video"
            )
            
            with gr.Row():
                start_btn = gr.Button("🚀 Start Inference", variant="primary")
                stop_btn = gr.Button("🛑 Stop")
            
            gr.Markdown("---")
            gr.Markdown("### 📊 Zone Analytics")
            zone_details = gr.Textbox(
                label="Live Occupancy & Directional Flow", 
                lines=15, 
                max_lines=20,
                interactive=False
            )

    # --- Event Wiring ---
    
    # 1. Start logic: Update status text, then trigger the generator
    start_btn.click(
        fn=start_pipeline, 
        inputs=[stream_input], 
        outputs=[status_label]
    ).then(
        fn=stream_vision, 
        outputs=[video_output, zone_details, alert_banner, fps_box]
    )
    
    # 2. Stop logic: Trigger stop and reset UI components
    stop_btn.click(
        fn=stop_pipeline, 
        outputs=[status_label, video_output, alert_banner, fps_box]
    )

if __name__ == "__main__":
    # Change share=True if you want to generate a public URL for the operator
    demo.launch(show_error=True)