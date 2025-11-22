"""
Extract frames from CCTV video at specified frame rate.
Usage:
  python scripts/extract_frames.py raw_video/street1.mp4 anonymized_frames/street1 [fps_sample]

If fps_sample is not provided, defaults to 0.1 (1 frame every 10 seconds).
"""
import cv2
import sys
from pathlib import Path

def extract_frames(video_path, output_dir, interval_sec=45):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"Total frames: {total_frames}")
    print(f"Original FPS: {original_fps:.2f}")
    print(f"Extracting 1 frame every {interval_sec} seconds")

    curr_time, saved = 0, 0
    next_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Get current video time in seconds:
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if frame_time >= next_capture_time:
            out_path = output_dir / f"frame_{saved:06d}.jpg"
            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                print(f"Warning: failed to save {out_path}")
            else:
                saved += 1
                next_capture_time += interval_sec
                if saved % 10 == 0:
                    print(f"Saved {saved} frames...", end='\r')

    cap.release()
    print(f"\nDone. Saved {saved} frames to {output_dir}")
    return saved


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/extract_frames.py <video_path> <output_dir> [interval_sec]")
        sys.exit(1)
    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    # Default to every 45 seconds
    interval_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0
    extract_frames(video_path, output_dir, interval_sec)
