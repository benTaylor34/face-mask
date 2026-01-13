import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from pathlib import Path


# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_VIDEO = BASE_DIR / "videos" / "test_vid.mp4"
TEMP_VIDEO = BASE_DIR / "outputs" / "temp_output.mp4"
FINAL_OUTPUT = BASE_DIR / "outputs" / "output_video.mp4"
MASK_IMG = BASE_DIR / "assets" / "mask.png"

FFMPEG_PATH = "ffmpeg" # r"C:\Users\Ben\Documents\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

MIN_CONFIDENCE = 0.7
DETECT_EVERY_N = 5
SMOOTHING_ALPHA = 0.7
MAX_MISSES = 15      # frames to tolerate before removing mask
# =========================================

# Load mask image (RGBA)
mask_img = cv2.imread(str(MASK_IMG), cv2.IMREAD_UNCHANGED)

# MediaPipe face detector
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=MIN_CONFIDENCE
)

# Video I/O
cap = cv2.VideoCapture(str(INPUT_VIDEO))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps != fps:#so it doesnt silently fail on windows
    fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(TEMP_VIDEO), fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("video writer failed to open")

# Tracking state
tracker = None
prev_bbox = None
last_valid_bbox = None
miss_count = 0
frame_idx = 0

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    use_detection = (frame_idx % DETECT_EVERY_N == 0) or tracker is None
    bbox_current = None

    # ---------- FACE DETECTION ----------
    if use_detection:
        results = face_detection.process(rgb)
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box

            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            # Expand to cover full head
            y = max(0, y - int(0.3 * h))
            h = int(h * 1.6)
            x = max(0, x - int(0.1 * w))
            w = int(w * 1.2)

            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))

            bbox_current = (x, y, w, h)
            miss_count = 0

    # ---------- TRACKING ----------
    elif tracker is not None:
        success, bbox = tracker.update(frame)
        if success:
            bbox_current = tuple(map(int, bbox))
            miss_count = 0
        else:
            miss_count += 1

    # ---------- MASK PERSISTENCE ----------
    if bbox_current is not None:
        last_valid_bbox = bbox_current
    elif last_valid_bbox is not None and miss_count < MAX_MISSES:
        bbox_current = last_valid_bbox
    else:
        tracker = None
        prev_bbox = None
        last_valid_bbox = None
        continue

    # ---------- TEMPORAL SMOOTHING ----------
    x, y, w, h = bbox_current

    if prev_bbox is not None:
        x = int(SMOOTHING_ALPHA * x + (1 - SMOOTHING_ALPHA) * prev_bbox[0])
        y = int(SMOOTHING_ALPHA * y + (1 - SMOOTHING_ALPHA) * prev_bbox[1])
        w = int(SMOOTHING_ALPHA * w + (1 - SMOOTHING_ALPHA) * prev_bbox[2])
        h = int(SMOOTHING_ALPHA * h + (1 - SMOOTHING_ALPHA) * prev_bbox[3])

    prev_bbox = (x, y, w, h)

    # Clamp to frame
    x = max(0, x)
    y = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    roi = frame[y:y2, x:x2]
    if roi.size > 0:
        mask = cv2.resize(mask_img, (roi.shape[1], roi.shape[0]))
        mask_rgb = mask[:, :, :3]
        mask_alpha = mask[:, :, 3] / 255.0
        mask_alpha = mask_alpha[:, :, None]

        roi[:] = (
            mask_alpha * mask_rgb +
            (1 - mask_alpha) * roi
        ).astype(np.uint8)

    out.write(frame)

# ================= CLEANUP =================
cap.release()
out.release()

# ================= FFMPEG COMPRESSION =================
ffmpeg_cmd = [
    str(FFMPEG_PATH),
    "-y",
    "-i", str(TEMP_VIDEO),
    "-c:v", "libx264",
    "-crf", "23",
    "-preset", "fast",
    str(FINAL_OUTPUT)
]

subprocess.run(ffmpeg_cmd)
os.remove(TEMP_VIDEO)

print("Finished. Output saved as:", FINAL_OUTPUT)

