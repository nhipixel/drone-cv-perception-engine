# Drone CV Perception Engine

A real-time computer vision framework developed as a Proof of Concept (POC) for autonomous drone-based reforestation using OpenCV. This system provides terrain analysis, microsite identification, and stability scoring to bridge the gap between aerial perception and physical seed deployment.

## Technical Overview

The perception engine utilizes a modular architecture to process high-resolution video streams. It identifies viable planting zones by segmenting substrate types and calculating a Confidence Score based on the UAV's spatial alignment with the target.

## Setup

1. **Clone the repository**
```bash
git clone [https://github.com/nhipixel/drone-cv-perception-engine.git](https://github.com/nhipixel/drone-cv-perception-engine.git)
```

2. **Create a virtual environment** 

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install numpy opencv-python
```

4. **Run the simulation**

```bash
python main_perception_engine.py
```

## Features
- **Substrate Segmentation**: Multi-layered HSV filtering to isolate plantable soil from vegetation and rocks.

- **Stability Scoring**: Heuristic-based centering analysis to optimize mechanical firing windows.

- **Telemetry HUD**: Real-time simulation of Ground Control Station (GCS) data including GPS, Altitude, and Battery.

- **Flight Logging**: Standardized logging of detection events for post-flight data analysis and Digital Identity mapping.
