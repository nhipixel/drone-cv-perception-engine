import cv2
import numpy as np
import logging
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MiON_CV] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Structured data for handoff to hardware team."""
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    confidence_score: float
    area_px: float

class MiON_Terrain_Simulator:
    """Generates synthetic high-fidelity terrain for CV validation."""
    @staticmethod
    def generate(width=800, height=600):
        # Base Soil
        terrain = np.zeros((height, width, 3), np.uint8)
        cv2.randn(terrain, (70, 80, 100), (15, 15, 15)) 
        
        # Viable Soil Patches (Targets)
        for _ in range(10):
            pt = (random.randint(50, 750), random.randint(50, 550))
            cv2.circle(terrain, pt, random.randint(40, 90), (45, 65, 95), -1)
        
        # Vegetation & Rocks (Obstacles)
        for _ in range(18):
            pt = (random.randint(0, 800), random.randint(0, 600))
            cv2.circle(terrain, pt, random.randint(20, 80), (40, 110, 45), -1)
        for _ in range(5):
            pts = np.array([[random.randint(0,800), random.randint(0,600)] for _ in range(3)])
            cv2.fillPoly(terrain, [pts], (90, 90, 95))

        return cv2.GaussianBlur(terrain, (5, 5), 0)

class MiON_CV_Processor:
    """Onboard Perception Engine for Autonomous Reforestation."""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "hsv_lower": np.array([5, 30, 30]),
            "hsv_upper": np.array([30, 160, 150]),
            "min_area": 2000,
            "max_area": 20000,
            "morph_kernel": 5
        }
        logger.info("MiON_CV Engine Initialized. Biome: Ontario_Coniferous_Mix.")

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        # Pre-processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config["hsv_lower"], self.config["hsv_upper"])
        kernel = np.ones((self.config["morph_kernel"], self.config["morph_kernel"]), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        f_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config["min_area"] < area < self.config["max_area"]:
                m = cv2.moments(cnt)
                if m['m00'] == 0: continue
                cx, cy = int(m['m10']/m['m00']), int(m['m01']/m['m00'])

                # Compute Stability Score (Inverse distance from lens center)
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(f_center))
                max_dist = np.linalg.norm(np.array([0, 0]) - np.array(f_center))
                score = round(max(0, (1 - (dist / max_dist)) * 100), 1)

                detections.append(DetectionResult((cx, cy), cv2.boundingRect(cnt), score, area))
        
        logger.info(f"Frame Processed. Detected Sites: {len(detections)}")
        return detections

class MiON_HUD_Visualizer:
    """Production HUD for Ground Control Station (GCS) monitoring."""
    @staticmethod
    def draw(frame: np.ndarray, detections: List[DetectionResult], telemetry: Dict):
        output = frame.copy()
        overlay = output.copy()
        
        # UI Background
        cv2.rectangle(overlay, (0, 0), (250, 600), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

        # Targets
        for det in detections:
            x, y, w, h = det.bounding_box
            color = (0, 255, 0) if det.confidence_score > 75 else (0, 180, 255)
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 1)
            cv2.drawMarker(output, det.centroid, color, cv2.MARKER_TILTED_CROSS, 15, 2)
            cv2.putText(output, f"{det.confidence_score}%", (x, y-8), 1, 0.7, color, 1)

        # Telemetry Text
        status_list = [
            "MiON_CV CORE v1.2",
            "----------------",
            f"OP: NHIYN",
            f"STATUS: {telemetry['status']}",
            f"SITES: {len(detections)}",
            f"BATT: {telemetry['batt']}",
            f"ALT: {telemetry['alt']}",
            f"GPS: {telemetry['gps']}",
            "----------------",
            "CMD: READY_TO_PLANT"
        ]
        for i, text in enumerate(status_list):
            cv2.putText(output, text, (20, 50 + i*35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)
        
        return output

# --- EXECUTION ---
if __name__ == "__main__":
    processor = MiON_CV_Processor()
    visualizer = MiON_HUD_Visualizer()
    
    raw_frame = MiON_Terrain_Simulator.generate()
    
    results = processor.process_frame(raw_frame)
    
    # Telemetry Data (Mocked from Onboard FCU)
    telemetry = {
        "status": "SEARCH_ACTIVE",
        "batt": "89.4%",
        "alt": "12.5m AGL",
        "gps": "43.257, -79.871", # Hamilton, ON
    }
    
    final_hud = visualizer.draw(raw_frame, results, telemetry)
    
    print("Simulation Running. Press any key on the image window to close.")
    cv2.imshow("MiON_CV Perception Terminal", final_hud)
    cv2.imwrite("MiON_CV_Final_Demo.jpg", final_hud)
    cv2.waitKey(0)
    cv2.destroyAllWindows()