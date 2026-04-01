# 🚀 Crowd Analytics System

A high-performance, multi-threaded pipeline for real-time person detection, tracking, and zone-based behavioral analytics. This system uses **YOLOv8** for detection, **ByteTrack** for persistent ID tracking, and a custom analytics engine to monitor occupancy and directional flow in specific regions of interest.

## ✨ Features
- **Real-time Detection & Tracking**: Powered by YOLOv8 and ByteTrack for robust per-person monitoring.
- **Zone-Based Analytics**: Define multiple polygons (zones) to monitor specific areas like entrances, crosswalks, or platforms.
- **Directional Flow**: Estimates the dominant movement direction (North, South-East, etc.) for people within each zone.
- **Automated Alerting**: Logs alerts to `logs/run.log` and structured JSON data when zone occupancy exceeds configurable thresholds.
- **Modern Web Console**: A premium Gradio-based interface for live monitoring and system control.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA/TensorRT support (recommended for real-time performance)
- Windows 10/11 (Optimized for `ProactorEventLoop`)

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/
cd crowd_analytics
pip install -r requirements.txt
```

> [!NOTE]
> Detailed requirements include: `ultralytics`, `gradio`, `opencv-python`, `shapely`, `pyyaml`, and `numpy`.

### 3. Model Weights
Ensure your fine-tuned model (e.g., `best.engine` or `best.pt`) is placed in the `weights/` directory:
- `weights/best.engine` (Default path used in `main.py` and `gradio_run.py`)

---

## 🚀 How to Run

### Option A: Standard Pipeline (OpenCV Window)
Ideal for testing and saving an annotated video locally.
```bash
python main.py
```
- **Output**: An annotated video will be saved to `outputs/final_annotated.mp4`.
- **Display**: Press `q` in the OpenCV window to stop early.

### Option B: Web Intelligence Console (Gradio)
The recommended way for operators to monitor the system live.
```bash
python gradio_run.py
```
1. Open the URL provided in the console (e.g., `http://127.0.0.1:7860`).
2. Select your **Input Source** (Sample Video or Webcam).
3. Click **🚀 Start Inference**.

---

## ⚙️ Configuration

### Configuring Zones & Thresholds
The system behavior is controlled via `configs/zones.yaml`. You can add, remove, or modify zones without touching the source code.

```yaml
zones:
  "Entrance_Alpha":
    coordinates: [[297, 522], [843, 549], [1198, 471], [958, 380]] # [x, y] polygon vertices
    threshold: 25 # Maximum person count before triggering an alert
```

- **Coordinates**: A list of `[x, y]` pairs representing the vertices of the polygon. Use the provided `utils/zone_drawer.py` tool to capture these coordinates from a frame.
- **Threshold**: The "Max Occupancy" for the zone. When the smoothed count exceeds this value, the zone turns **Red** and an alert is logged.

### Logs & Data
- **Run Logs**: `logs/run.log` contains human-readable status updates and alerts.
- **Structured Alerts**: `logs/alerts.jsonl` contains machine-readable JSON alert events for further integration.

---

## 🏗️ System Architecture
The system uses a **Parallelized Three-Thread Pipeline** to maximize throughput:
1. **VideoIngestor**: Handles I/O-bound frame reading.
2. **Inferencer**: Compute-bound YOLOv8+ByteTrack processing.
3. **Analytics**: Business logic, zone calculations, and smoothing.
4. **Main Thread**: Handles UI rendering and event loop management.

This architecture ensures that inference latency doesn't block video reading, and the UI remains responsive even under heavy load.