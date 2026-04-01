from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any

@dataclass
class FramePayload:
    frame_id: int
    image: np.ndarray
    tracks: List[Dict[str, Any]] = field(default_factory=list) # Holds ID, box, centroid
    zone_counts: Dict[str, int] = field(default_factory=dict)  # Smoothed counts
    zone_flows: Dict[str, str] = field(default_factory=dict)   # E.g., {"Platform A": "North-West"}
    alerts: List[str] = field(default_factory=list)