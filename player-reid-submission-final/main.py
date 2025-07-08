# Player Re-Identification Assignment for Liat.ai
# --------------------------------------------------
# Author: E PRASHANTH REDDY
# Submission: 

import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort





VIDEO_BROADCAST = 'videos/broadcast.mp4'
VIDEO_TACTICAM = 'videos/tacticam.mp4'
MODEL_PATH = 'models/yolov11_player_ball.pt'  


os.makedirs("outputs", exist_ok=True)


model = YOLO(models/yolov11_player_ball.pt)

tracker = DeepSort(max_age=30)



def process_video(video_path, output_path, camera_label):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)[0]

        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) == 0:  
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{camera_label}_ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(f"{output_path}/frame_{frame_count:05d}.jpg", frame)

    cap.release()
    print(f"Done processing {video_path}")


# Run pipeline on both videos
process_video(VIDEO_BROADCAST, "outputs/broadcast", "B")
process_video(VIDEO_TACTICAM, "outputs/tacticam", "T")

def extract_features(image):
    image = cv2.resize(image, (64, 128))
    return image.flatten() / 255.0


def match_players(folder1, folder2):
    features1 = {}
    features2 = {}

    for f in sorted(os.listdir(folder1))[:30]:
        img = cv2.imread(os.path.join(folder1, f))
        features1[f] = extract_features(img)

    for f in sorted(os.listdir(folder2))[:30]:
        img = cv2.imread(os.path.join(folder2, f))
        features2[f] = extract_features(img)

    print("Matching Players Across Cameras:")
    for k1, v1 in features1.items():
        best_match = None
        best_score = float("inf")
        for k2, v2 in features2.items():
            dist = np.linalg.norm(v1 - v2)
            if dist < best_score:
                best_score = dist
                best_match = k2
        print(f"{k1} matches with {best_match} (score={best_score:.4f})")

match_players("outputs/broadcast", "outputs/tacticam")
