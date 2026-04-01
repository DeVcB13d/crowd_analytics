import os
import time
import queue
import threading
import cv2
import numpy as np
import logging
import pandas as pd
import json
import yaml
import argparse
from datetime import datetime
from typing import List, Dict, Any

# OPENMP Windows fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser(description="Crowd Analytics Gradio Interface")
    parser.add_argument("--config", type=str, default="configs/zones.yaml", help="Path to zones configuration")
    parser.add_argument("--model", type=str, default="weights/best.engine", help="Path to YOLO model engine")
    parser.add_argument("--video", type=str, default="samples/sample_video.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/final_annotated.mp4", help="Path to output annotated video (Note: not used in live stream)")
    parser.add_argument("--log", type=str, default="logs/alerts.jsonl", help="Path to alerts log file")
    return parser.parse_known_args()[0]

args = parse_args()

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
        self.visualizer = Visualizer(args.config)
        
        # Load thresholds directly from config to avoid repeating logic
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            self.thresholds = {name: data["threshold"] for name, data in config.get("zones", {}).items()}
        except Exception as e:
            print(f"Failed to load thresholds from {args.config}: {e}")
            self.thresholds = {}

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
        cv2.destroyAllWindows()

manager = PipelineManager()

# --- Logic Functions ---

def start_pipeline(video_choice):
    """Initializes and kicks off the background workers."""
    manager.stop_event.clear()
    
    # Path Mapping
    video_path = args.video if "Sample" in video_choice else 0
    
    # Internal Queues
    frame_q = queue.Queue(maxsize=30)
    detect_q = queue.Queue(maxsize=30)
    
    # Instantiate Workers (Using your existing classes)
    ingestor = VideoIngestor(video_path=video_path, output_queue=frame_q)
    inference = Inference(model_path=args.model, input_queue=frame_q, output_queue=detect_q)
    analytics = Analytics(config_path=args.config, input_queue=detect_q, output_queue=manager.output_queue)

    # Start Threads
    manager.threads = [ingestor, inference, analytics]
    for t in manager.threads:
        t.daemon = True # Ensure threads exit if main process exits
        t.start()
    
    return "Pipeline Status: RUNNING"

def stream_vision():
    """Generator that yields data from the queue to the Gradio UI."""
    prev_time = time.time()
    start_time = time.time()
    
    # State tracking
    history_data = []
    log_messages = []
    total_alerts = 0
    peak_count = 0
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log_file_path = args.log
    
    # Default values for outputs when waiting
    empty_df = pd.DataFrame(columns=["time", "count"])
    df_plot = empty_df
    metrics_html = "Waiting for stream..."
    bars_html = "Waiting for zone data..."
    logs_html = "Waiting for events..."

    try:
        while not manager.stop_event.is_set():
            try:
                # Wait for a payload with a short timeout to keep the loop responsive
                payload: FramePayload = manager.output_queue.get(timeout=0.1)
                
                # 1. Process Image & Annotate
                annotated_frame = manager.visualizer.draw(payload)
                
                # OPTIMIZATION: Downscale to fit standard screens
                h, w = annotated_frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    annotated_frame = cv2.resize(annotated_frame, (1280, int(h * scale)))

                cv2.imshow("Crowd Analytics Live Feed", annotated_frame)
                cv2.waitKey(1)
                
                # 2. Performance Tracking
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                # Metrics Extraction
                total_count = sum(payload.zone_counts.values())
                peak_count = max(peak_count, total_count)
                
                # Update Sparkline Data
                elapsed_time = curr_time - start_time
                history_data.append({"time": elapsed_time, "count": total_count})
                # Keep only last 30 seconds
                history_data = [d for d in history_data if elapsed_time - d["time"] <= 30]
                df_plot = pd.DataFrame(history_data) if history_data else empty_df
                
                # Update Ops Panel HTML: 4 Metric Cards
                metrics_html = f"""
                <div style="display: flex; justify-content: space-between; text-align: center; background: #222; padding: 10px; border-radius: 8px;">
                    <div style="flex: 1;">
                        <span style="font-size: 14px; color: #aaa; text-transform: uppercase;">Alerts</span><br/>
                        <span style="font-size: 28px; font-weight: bold; color: {'#ff4a4a' if total_alerts > 0 else '#fff'};">{total_alerts}</span>
                    </div>
                    <div style="flex: 1;">
                        <span style="font-size: 14px; color: #aaa; text-transform: uppercase;">Total</span><br/>
                        <span style="font-size: 28px; font-weight: bold; color: #fff;">{total_count}</span>
                    </div>
                    <div style="flex: 1;">
                        <span style="font-size: 14px; color: #aaa; text-transform: uppercase;">Peak</span><br/>
                        <span style="font-size: 28px; font-weight: bold; color: #fff;">{peak_count}</span>
                    </div>
                    <div style="flex: 1;">
                        <span style="font-size: 14px; color: #aaa; text-transform: uppercase;">FPS</span><br/>
                        <span style="font-size: 28px; font-weight: bold; color: #fff;">{fps:.1f}</span>
                    </div>
                </div>
                """
                
                # Update Ops Panel HTML: Per-zone Occupancy Bars
                bars_html = '<div style="background: #222; padding: 15px; border-radius: 8px;">'
                bars_html += '<h3 style="margin-top:0; color:#fff; font-size:16px;">Zone Occupancy</h3>'
                for zone, count in payload.zone_counts.items():
                    threshold = manager.thresholds.get(zone, 20)
                    pct = min(count / threshold * 100, 100) if threshold > 0 else 0
                    
                    if count >= threshold:
                        color = "#ff4a4a" # critical
                    elif count >= threshold * 0.75:
                        color = "#ffae42" # warn
                    else:
                        color = "#4caf50" # safe
                        
                    bars_html += f"""
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="font-weight: bold; color: #eee; font-size: 14px;">{zone}</span>
                        </div>
                        <div style="background-color: #444; width: 100%; height: 10px; border-radius: 5px; overflow: hidden;">
                            <div style="background-color: {color}; width: {pct}%; height: 100%; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                    """
                bars_html += '</div>'
                
                # Log JSONL stream and Handle Alerts
                if payload.alerts:
                    for alert_msg in payload.alerts:
                        zone_name = "System"
                        for z in manager.thresholds.keys():
                            if z in alert_msg:
                                zone_name = z
                                break
                        
                        log_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "zone_name": zone_name,
                            "event": "threshold_exceeded",
                            "message": alert_msg
                        }
                        
                        # Write to JSONL file
                        with open(log_file_path, 'a') as f:
                            f.write(json.dumps(log_entry) + '\\n')
                        
                        # Add to UI logs memory (keep last 50)
                        total_alerts += 1
                        badge = '<span style="background:#ff4a4a; color:white; padding:2px 6px; border-radius:3px; font-size:12px; font-weight:bold;">CRITICAL</span>'
                        time_str = datetime.now().strftime("%H:%M:%S")
                        log_messages.insert(0, f"[{time_str}] {badge} {alert_msg}")
                        
                log_messages = log_messages[:50]
                
                logs_html = f"""
                <div style="background: #111; color: #0f0; font-family: monospace; padding: 10px; border-radius: 8px; height: 250px; overflow-y: auto; font-size: 13px;">
                    {"<br/>".join(log_messages) if log_messages else "Waiting for events..."}
                </div>
                """
                
                yield (
                    metrics_html,
                    bars_html,
                    df_plot,
                    logs_html,
                    "Pipeline Status: RUNNING"
                )
                
                # OPTIONAL: Reduced Throttling
                time.sleep(0.01) 

            except queue.Empty:
                # If the queue is empty, just yield the last state to keep the UI alive
                yield (
                    metrics_html,
                    bars_html,
                    df_plot if 'df_plot' in locals() else empty_df,
                    logs_html,
                    "Pipeline Status: RUNNING (Waiting...)"
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
    empty_df = pd.DataFrame(columns=["time", "count"])
    metrics = "Pipeline Stopped."
    bars = "Pipeline Stopped."
    logs = "Pipeline Stopped."
    return metrics, bars, empty_df, logs, "Pipeline Status: STOPPED"

# --- Gradio UI Definition ---

with gr.Blocks(title="VariPhi AI Console") as demo:
    gr.Markdown("# Crowd Analytics")
    
    with gr.Row():
        # Left Side: Controls
        with gr.Column(scale=3):
            gr.Markdown("### Application Controls")
            status_label = gr.Markdown("Pipeline Status: IDLE")
            
            stream_input = gr.Dropdown(
                choices=["Sample Video", "Webcam / Live Stream"], 
                label="Select Input Source", 
                value="Sample Video"
            )
            
            with gr.Row():
                start_btn = gr.Button("🚀 Start Inference", variant="primary")
                stop_btn = gr.Button("🛑 Stop")
            
            gr.Markdown("---")
            gr.Markdown("> *Note: The Live Video Feed will pop up in a dedicated OpenCV Window.*")
            
        # Right Side: Live Ops Dashboard
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Live Ops Dashboard")
            
            # 1. Metric Cards
            metrics_html_component = gr.HTML(value="""
            <div style="display: flex; justify-content: space-between; text-align: center; background: #222; padding: 10px; border-radius: 8px;">
                <div style="flex: 1;"><span style="font-size: 14px; color: #aaa;">ALERTS</span><br/><span style="font-size: 28px; color: #fff;">0</span></div>
                <div style="flex: 1;"><span style="font-size: 14px; color: #aaa;">TOTAL</span><br/><span style="font-size: 28px; color: #fff;">0</span></div>
                <div style="flex: 1;"><span style="font-size: 14px; color: #aaa;">PEAK</span><br/><span style="font-size: 28px; color: #fff;">0</span></div>
                <div style="flex: 1;"><span style="font-size: 14px; color: #aaa;">FPS</span><br/><span style="font-size: 28px; color: #fff;">0.0</span></div>
            </div>
            """)
            
            # 2. Occupancy Bars
            bars_html_component = gr.HTML(value="""
            <div style="background: #222; padding: 15px; border-radius: 8px;">
                <h3 style="margin-top:0; color:#fff; font-size:16px;">Zone Occupancy</h3>
                <p style="color:#aaa;">Waiting for data...</p>
            </div>
            """)
            
            # 3. Sparkline Plot
            empty_df = pd.DataFrame(columns=["time", "count"])
            plot_component = gr.LinePlot(
                value=empty_df, 
                x="time", 
                y="count", 
                title="Crowd Trajectory (30s Rolling)", 
                height=250
            )
            
            # 4. JSONL Stream Viewer
            gr.Markdown("### JSONL Event Stream")
            logs_html_component = gr.HTML(value="""
                <div style="background: #111; color: #0f0; font-family: monospace; padding: 10px; border-radius: 8px; height: 250px; overflow-y: auto; font-size: 13px;">
                    Waiting for events...
                </div>
            """)

    # --- Event Wiring ---
    
    # 1. Start logic: Update status text, then trigger the generator
    start_btn.click(
        fn=start_pipeline, 
        inputs=[stream_input], 
        outputs=[status_label]
    ).then(
        fn=stream_vision, 
        outputs=[metrics_html_component, bars_html_component, plot_component, logs_html_component, status_label]
    )
    
    # 2. Stop logic: Trigger stop and reset UI components
    stop_btn.click(
        fn=stop_pipeline, 
        outputs=[metrics_html_component, bars_html_component, plot_component, logs_html_component, status_label]
    )

if __name__ == "__main__":
    # Change share=True if you want to generate a public URL for the operator
    demo.launch(
        show_error=True, 
        theme=gr.themes.Default(primary_hue="orange")
    )