# Player Re-Identification Assignment

This project solves a real-world computer vision problem of re-identifying players across multiple camera views using object detection and tracking.

## Tasks Done
- Player detection using YOLOv11
- Tracking using Deep SORT
- Re-identification using appearance features

## Requirements
- Python 3.8+
- OpenCV
- Ultralytics YOLOv11
- deep_sort_realtime

## How to Run
1. Place `broadcast.mp4` and `tacticam.mp4` in `videos/` folder.
2. Place the YOLOv11 model in `models/`.
3. Run the script: `python main.py`
4. Check output in the `outputs/` folder.

## Credits
- Ultralytics YOLO
- Deep SORT
