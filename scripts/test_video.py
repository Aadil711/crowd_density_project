import cv2
import sys
from pathlib import Path

# Check if video file exists
video_path = "raw_video/street1.mp4"

if not Path(video_path).exists():
    print(f"❌ ERROR: Video file not found at: {video_path}")
    print(f"   Current working directory: {Path.cwd()}")
    print(f"   Please check:")
    print(f"   1. Is the video file in the raw_video/ folder?")
    print(f"   2. Is the filename spelled correctly?")
    sys.exit(1)

print(f"✅ Video file found: {video_path}")
print(f"   File size: {Path(video_path).stat().st_size / (1024*1024):.2f} MB")

# Try to open the video
print("\n🔄 Attempting to open video...")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ERROR: OpenCV cannot open this video file")
    print("   Possible reasons:")
    print("   1. Video codec not supported")
    print("   2. File is corrupted")
    print("   3. Wrong file format")
    sys.exit(1)

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps if fps > 0 else 0

print("✅ Video opened successfully!")
print(f"   Resolution: {width}x{height}")
print(f"   FPS: {fps:.2f}")
print(f"   Total frames: {total_frames}")
print(f"   Duration: {duration:.2f} seconds")

# Try to read first frame
ret, frame = cap.read()
if ret:
    print("✅ Successfully read first frame!")
    print(f"   Frame shape: {frame.shape}")
else:
    print("❌ ERROR: Could not read first frame")

cap.release()
print("\n✅ All checks passed! Your video should work.")
