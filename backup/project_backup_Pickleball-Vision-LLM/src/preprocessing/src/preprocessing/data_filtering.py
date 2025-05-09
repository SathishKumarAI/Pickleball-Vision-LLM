import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict
from tqdm import tqdm
import argparse
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import imagehash
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_blurry(frame: np.ndarray, threshold: float) -> bool:
    """Check if a frame is blurry using the Laplacian variance method."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def is_duplicate(frame1: np.ndarray, frame2: np.ndarray, hash_size: int = 8) -> bool:
    """Check if two frames are duplicates using perceptual hashing."""
    hash1 = imagehash.average_hash(Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)), hash_size=hash_size)
    hash2 = imagehash.average_hash(Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)), hash_size=hash_size)
    return hash1 == hash2

def has_motion(frame1: np.ndarray, frame2: np.ndarray, threshold: float) -> bool:
    """Check if there is significant motion between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    non_zero_count = np.count_nonzero(diff)
    return non_zero_count > threshold

def is_within_roi(frame: np.ndarray, roi: Dict[str, int]) -> bool:
    """Check if the frame contains content within the region of interest (ROI)."""
    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    cropped = frame[y:y+h, x:x+w]
    return cropped.size > 0

def filter_frames(frames: List[np.ndarray], config: dict) -> List[np.ndarray]:
    """
    Filter frames based on various criteria.
    
    Args:
        frames (List[np.ndarray]): List of video frames.
        config (dict): Configuration dictionary with thresholds and ROI.

    Returns:
        List[np.ndarray]: Filtered frames.
    """
    filtered_frames = []
    prev_frame = None

    for i, frame in enumerate(tqdm(frames, desc="Filtering frames")):
        if is_blurry(frame, config["blurriness_threshold"]):
            logging.info(f"Frame {i} filtered out: blurry")
            continue

        if prev_frame is not None and is_duplicate(prev_frame, frame):
            logging.info(f"Frame {i} filtered out: duplicate")
            continue

        if prev_frame is not None and not has_motion(prev_frame, frame, config["motion_threshold"]):
            logging.info(f"Frame {i} filtered out: minimal motion")
            continue

        if not is_within_roi(frame, config["roi"]):
            logging.info(f"Frame {i} filtered out: outside ROI")
            continue

        filtered_frames.append(frame)
        prev_frame = frame

    return filtered_frames

def extract_frames(video_path: str) -> List[np.ndarray]:
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_frames(frames: List[np.ndarray], output_folder: str):
    """Save filtered frames to the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_folder, f"frame_{i:04d}.jpg")
        cv2.imwrite(output_path, frame)

def main():
    parser = argparse.ArgumentParser(description="Filter video frames based on quality and motion.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video or folder of videos.")
    parser.add_argument("--output", type=str, required=True, help="Path to save filtered frames.")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file (JSON).")
    args = parser.parse_args()

    import json
    with open(args.config, "r") as f:
        config = json.load(f)

    if os.path.isfile(args.input):
        videos = [args.input]
    else:
        videos = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith((".mp4", ".avi"))]

    for video_path in videos:
        logging.info(f"Processing video: {video_path}")
        frames = extract_frames(video_path)
        filtered_frames = filter_frames(frames, config)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(args.output, video_name)
        save_frames(filtered_frames, output_folder)

if __name__ == "__main__":
    main()