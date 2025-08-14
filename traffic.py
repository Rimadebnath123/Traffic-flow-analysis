import os
import sys
import time
import math
import argparse
from pathlib import Path
from datetime import timedelta

import cv2
import numpy as np
import pandas as pd

# Detector: YOLOv8 (Ultralytics)
from ultralytics import YOLO

# Tracker: DeepSORT (deep-sort-realtime)
from deep_sort_realtime.deepsort_tracker import DeepSort

# Optional: YouTube download (pytube)
try:
    from pytube import YouTube
except Exception:
    YouTube = None


# -----------------------------
# Utility functions
# -----------------------------
COCO_VEHICLE_CLASS_IDS = {2, 3, 5, 7}  
# 2 car, 3 motorcycle, 5 bus, 7 truck (YOLO/COCO indexing commonly used by Ultralytics)

def download_youtube_video(url: str, out_dir: str) -> str:
    """
    Downloads the highest progressive MP4 stream if pytube is available.
    Returns path to downloaded file.
    """
    if YouTube is None:
        raise RuntimeError("pytube is not installed. Install it or provide a local video path.")
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    if stream is None:
        raise RuntimeError("No suitable MP4 progressive stream found.")
    os.makedirs(out_dir, exist_ok=True)
    fp = stream.download(output_path=out_dir, filename="traffic_source.mp4")
    return fp


def lane_index_for_x(x, w, lane_splits=(1/3, 2/3)):
    """
    Assign lane by x coordinate and frame width.
    lane_splits are fractional boundaries (e.g., 1/3 and 2/3).
    Returns 1, 2, or 3 (lane numbers).
    """
    b1 = int(w * lane_splits[0])
    b2 = int(w * lane_splits[1])
    if x < b1:
        return 1
    elif x < b2:
        return 2
    else:
        return 3


def draw_lane_bands(frame, lane_splits=(1/3, 2/3)):
    """
    Draws three vertical bands to visualize lanes.
    """
    h, w = frame.shape[:2]
    b1, b2 = int(w * lane_splits[0]), int(w * lane_splits[1])

    overlay = frame.copy()
    # Light transparent overlays for lanes
    cv2.rectangle(overlay, (0, 0), (b1, h), (255, 255, 255), -1)
    cv2.rectangle(overlay, (b1, 0), (b2, h), (255, 255, 255), -1)
    cv2.rectangle(overlay, (b2, 0), (w, h), (255, 255, 255), -1)
    alpha = 0.08
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Lane boundary lines
    cv2.line(frame, (b1, 0), (b1, h), (255, 255, 255), 2)
    cv2.line(frame, (b2, 0), (b2, h), (255, 255, 255), 2)

    # Lane labels
    cv2.putText(frame, "Lane 1", (int(b1*0.33), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Lane 2", (int(b1 + (b2-b1)*0.33), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Lane 3", (int(b2 + (w-b2)*0.33), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Traffic Flow Analysis (3-lane counting with tracking)")
    parser.add_argument("--source", type=str, default="", help="Path to local video. If empty, will try to download from YouTube URL.")
    parser.add_argument("--yt_url", type=str, default="https://www.youtube.com/watch?v=MNn9qKG2UFI", help="YouTube URL (if --source not provided).")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Folder for outputs (video + CSV).")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics YOLOv8 model weights.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--line_pos", type=float, default=0.60, help="Horizontal count line as fraction of height (0-1).")
    parser.add_argument("--lane_b1", type=float, default=1/3, help="First vertical split (0-1).")
    parser.add_argument("--lane_b2", type=float, default=2/3, help="Second vertical split (0-1).")
    parser.add_argument("--save_video", action="store_true", help="Save annotated video to outputs/annotated.mp4")
    parser.add_argument("--show", action="store_true", help="Show live window")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve source
    video_path = args.source
    if not video_path:
        print("No --source provided; attempting YouTube download...")
        video_path = download_youtube_video(args.yt_url, args.out_dir)
        print(f"Downloaded: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: cannot open video source.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Count line position
    count_line_y = int(height * args.line_pos)
    lane_splits = (args.lane_b1, args.lane_b2)

    # Initialize detector and tracker
    model = YOLO(args.model)  # downloads weights on first use

    tracker = DeepSort(
        max_age=30,        # frames to keep "lost" tracks
        n_init=2,          # dets before confirming a track
        max_iou_distance=0.7,
        max_cosine_distance=0.3,  # default metric
        nn_budget=None,
    )

    # Counting state
    lane_counts = {1: 0, 2: 0, 3: 0}
    counted_ids = set()   # IDs that have crossed the line
    csv_rows = []         # to write results

    # Video writer (optional)
    out_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            os.path.join(args.out_dir, "annotated.mp4"),
            fourcc, fps, (width, height)
        )

    start_time = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Inference
            results = model.predict(
                source=frame, conf=args.conf, iou=args.iou, verbose=False
            )

            dets_for_tracker = []
            if len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), cf, c in zip(xyxy, confs, clss):
                        if c in COCO_VEHICLE_CLASS_IDS:
                            dets_for_tracker.append(([x1, y1, x2, y2], float(cf), c))

            # Track update: expects [ [x1,y1,x2,y2], conf, class ] per detection
            tracks = tracker.update_tracks(dets_for_tracker, frame=frame)

            # Draw lanes and count line
            frame = draw_lane_bands(frame, lane_splits=lane_splits)
            cv2.line(frame, (0, count_line_y), (width, count_line_y), (0, 255, 255), 2)
            cv2.putText(frame, "Count Line", (10, count_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # Handle tracks
            for t in tracks:
                if not t.is_confirmed():
                    continue
                track_id = t.track_id
                ltrb = t.to_ltrb()  # left, top, right, bottom
                x1, y1, x2, y2 = map(int, ltrb)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Draw box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # Count when centroid crosses the line for the first time
                if track_id not in counted_ids and cy >= count_line_y:
                    lane = lane_index_for_x(cx, width, lane_splits=lane_splits)
                    lane_counts[lane] += 1
                    counted_ids.add(track_id)

                    # Timestamp from frame index
                    timestamp_sec = frame_idx / fps
                    csv_rows.append({
                        "vehicle_id": track_id,
                        "lane": lane,
                        "frame": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 3)
                    })

            # Live lane counters
            y0 = 60
            for lane_num in (1, 2, 3):
                txt = f"Lane {lane_num}: {lane_counts[lane_num]}"
                cv2.putText(frame, txt, (10, y0 + 30 * (lane_num - 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
                cv2.putText(frame, txt, (10, y0 + 30 * (lane_num - 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            # Write / show
            if out_writer is not None:
                out_writer.write(frame)

            if args.show:
                cv2.imshow("Traffic Flow Analysis", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

        # Save CSV
        csv_path = os.path.join(args.out_dir, "counts.csv")
        pd.DataFrame(csv_rows, columns=["vehicle_id", "lane", "frame", "timestamp_sec"]).to_csv(csv_path, index=False)

        # Summary
        summary = f"\n=== SUMMARY ===\n" \
                  f"Lane 1: {lane_counts[1]}\n" \
                  f"Lane 2: {lane_counts[2]}\n" \
                  f"Lane 3: {lane_counts[3]}\n" \
                  f"Total : {lane_counts[1] + lane_counts[2] + lane_counts[3]}\n"
        print(summary)

        print(f"CSV saved to: {csv_path}")
        if out_writer is not None:
            print(f"Annotated video saved to: {os.path.join(args.out_dir, 'annotated.mp4')}")

    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
