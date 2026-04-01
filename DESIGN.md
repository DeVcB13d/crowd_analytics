# System Design 

## Inference Pipeline

The pipeline is designed as different paralelly running components communicating through `Queue objects`. Each thread does exactly one job and communicates only through `Queue objects`. A `FramePayload` dataclass starts bare (just the raw frame) and gets fields filled in as it moves through the pipeline. No thread talks to another directly if a thread is slow, the queue fills up and the upstream thread blocks automatically. The multi-thread queue architecture separates I/O-bound work (reading frames) from compute-bound work (inference) so neither blocks the other.

![System Flow](docs/images/Video%20Ingestion%20to-2026-04-01-152012.png)

Thread responsibilities, clearly
    *VideoIngestor (Thread 1) — Reads the video file and creates FramePayload objects. It puts them into the `frame_queue`. When the video ends, it puts None to signal everyone downstream. In case of a live stream this thread can be replaced with a `WebcamIngestor` thread.
    *Inferencer (Thread 2) — pulls from frame_queue, runs `model.track()`, attaches the bounding boxes and ByteTrack IDs to the payload, then puts it into `detection_queue`. 
    *Analytics (Thread 3) — pulls from detection_queue and does all the analysis logic: checking which zone each tracked person's centroid falls inside, computing optical flow vectors per zone, applying rolling average smoothing, and firing alerts to the log file when thresholds are crossed. Puts the enriched payload into `output_queue`.
    *Renderer (Thread 4) — This thread is responsible for rendering the output
    It pulls from `output_queue` and draws zone overlays, counts, flow direction arrows, and alert flashes onto the frame.



## Detection Approach 

For detection, state of the art model yoloV8 is fine-tuned on the Crowdhuman Dataset for the purpose of person detection. Eventhough a density map based approach can give a better accuraccy in highly crowded scenarios, a detection based approach was used as it would be more robust with the optical flow approach and can be used along with ByteTrack to do per-person flow estimation. The approach is more versatile, as it could be made better in the future by adding elements of the density map.

## Output & Logging 


