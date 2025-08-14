# Traffic Flow Analysis (3 Lanes, YOLOv8 + DeepSORT)

This project counts vehicles per lane in a traffic video using YOLOv8 (COCO) for detection and DeepSORT for tracking. It prevents double counting by counting each track exactly once when crossing a virtual line.

## Features
- Pretrained YOLOv8 vehicle detection (car/motorcycle/bus/truck)
- DeepSORT tracking for stable IDs across frames
- Three-lane definition via vertical splits (tunable)
- Real-time or near real-time performance based on your hardware
- CSV export (vehicle_id, lane, frame, timestamp_sec)
- Annotated video with lane overlays and live counters
- Summary counts per lane

## Setup
creat a file in vs code using python extensions


pip install --upgrade pip
pip install ultralytics opencv-python numpy pytube deep-sort-realtime pandas
