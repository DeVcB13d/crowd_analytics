# System Design 

## Inference Pipeline

The pipeline is designed as different paralelly running components communicating through `Queue objects`. Each thread does exactly one job and communicates only through `Queue objects`. A `FramePayload` dataclass starts bare (just the raw frame) and gets fields filled in as it moves through the pipeline. No thread talks to another directly if a thread is slow, the queue fills up and the upstream thread blocks automatically. The multi-thread queue architecture separates I/O-bound work (reading frames) from compute-bound work (inference) so neither blocks the other.

![System Flow](docs/images/Video%20Ingestion%20to-2026-04-01-152012.png)

* *VideoIngestor (Thread 1)* — Reads the video file and creates FramePayload objects. It puts them into the `frame_queue`. When the video ends, it puts None to signal everyone downstream. In case of a live stream this thread can be replaced with a `WebcamIngestor` thread.
* *Inferencer (Thread 2)* — pulls from frame_queue, runs `model.track()`, attaches the bounding boxes and ByteTrack IDs to the payload, then puts it into `detection_queue`. 
* *Analytics (Thread 3)* — pulls from detection_queue and does all the analysis logic: checking which zone each tracked person's centroid falls inside, computing optical flow vectors per zone, applying rolling average smoothing, and firing alerts to the log file when thresholds are crossed. Puts the enriched payload into `output_queue`.
* *Renderer (Thread 4)* — This thread is responsible for rendering the output
It pulls from `output_queue` and draws zone overlays, counts, flow direction, and alert flashes onto the frame.

## Zone Logic & Flow Estimation

The system initializes from the YAML definition of at least three named polygonal zones and their respective occupancy limits. These coordinates are converted into Shapely Polygon objects to enable spatial calculations,the population limits are stored in a dictionary called self.thresholds for quick lookups during the processing loop. For every frame processed in the run method, the code checks if the centroid of each tracked person is located within a zone's polygon To ensure data stability, the system maintains a `count_history` using a deque to store the last 15 frames of raw counts, calculating a *rolling average* to smooth out potential noise. If this smoothed count exceeds the configured threshold for a specific zone, the system automatically appends a text alert to the output payload.
A velocity vector is computed based on the displacement between the earliest and latest coordinates.  Within each zone,these individual vectors are averaged to find the dominant movement direction and passes the result to get direction. 

## Detection Approach 

For detection, state of the art model yoloV8 is fine-tuned on the Crowdhuman Dataset for the purpose of person detection. Eventhough a density map based approach can give a better accuraccy in highly crowded scenarios, a detection based approach was used as it would be more robust with the optical flow approach and can be used along with ByteTrack to do per-person flow estimation. The approach is more versatile, as it could be made better in the future by adding elements of the density map.

## Output & Logging 

I designed a Live Ops Dashboard to make the system genuinely usable in real-world monitoring scenarios. The interface surfaces key information instantly through a compact 4-metric grid (alerts, total count, peak, FPS) and color-coded occupancy bars, making system status understandable at a glance. A real-time JSONL event viewer with severity indicators enables immediate auditing of threshold breaches without switching contexts. 

