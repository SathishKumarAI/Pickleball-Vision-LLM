import os
import logging
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed  # Added this import
# -----------------------------#
# Configuration
# -----------------------------#
VIDEO_DIR = Path("data/raw_videos")
FRAME_DIR = Path("data/frames")
LOG_PATH = Path("logs/frame_sampler.log")

# Ensure directories exist
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------#
# Utilities
# -----------------------------#
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract frames from videos at specified intervals.")
    parser.add_argument("--frame-rate", type=int, default=1, help="Interval in seconds to extract frames.")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of worker threads for parallel processing.")
    parser.add_argument("--video-dir", type=str, default=str(VIDEO_DIR), help="Directory where videos are stored.")
    parser.add_argument("--frame-dir", type=str, default=str(FRAME_DIR), help="Directory to save extracted frames.")
    return parser.parse_args()

def is_valid_video(path):
    """Check if the given path is a valid video file."""
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}

def sanitize_filename(filename):
    """Sanitize filenames by removing illegal characters."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

# -----------------------------#
# Frame Extraction
# -----------------------------#
def extract_frames(video_path, frame_rate=1):
    """Extract frames from a video at the specified frame rate (in seconds)."""
    video_name = video_path.stem
    frames_saved = []

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in video
        interval = int(fps * frame_rate)  # Frame interval based on the desired rate

        logging.info(f"Extracting frames from {video_name} at {frame_rate}s intervals.")
        
        frame_count = 0
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame at the specified interval
            if frame_count % interval == 0:
                frame_filename = FRAME_DIR / f"{video_name}_frame_{frame_id:04d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                frames_saved.append(str(frame_filename))
                frame_id += 1

            frame_count += 1

        cap.release()
        logging.info(f"Extracted {len(frames_saved)} frames from {video_name}.")
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {str(e)}")

    return frames_saved

# -----------------------------#
# Main Execution Flow
# -----------------------------#
def process_video(video_path, frame_rate):
    """Process a single video and extract frames."""
    if not is_valid_video(video_path):
        logging.warning(f"Skipping invalid video file: {video_path}")
        return []

    logging.info(f"Started processing video: {video_path}")
    frames = extract_frames(video_path, frame_rate)
    logging.info(f"Completed processing video: {video_path}")
    return frames

def process_videos_parallel(video_files, frame_rate, max_workers):
    """Process multiple videos in parallel using ThreadPoolExecutor."""
    all_frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_video, video, frame_rate): video for video in video_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
            frames = future.result()
            all_frames.extend(frames)

    return all_frames

# -----------------------------#
# Main Program Logic
# -----------------------------#
def main():
    args = parse_args()

    # Ensure paths are correct
    global VIDEO_DIR, FRAME_DIR
    VIDEO_DIR = Path(args.video_dir)
    FRAME_DIR = Path(args.frame_dir)

    # Validate video directory
    if not VIDEO_DIR.exists():
        logging.error(f"Video directory {VIDEO_DIR} does not exist.")
        print("❌ Video directory not found.")
        return
    else:
        print(f"✔ Video directory found: {VIDEO_DIR}")

    print("🔍 Searching for downloaded videos...")
    video_files = [f for f in VIDEO_DIR.glob("*") if is_valid_video(f)]
    
    # Debugging step - ensure video files are detected
    if not video_files:
        logging.error(f"No valid video files found in {VIDEO_DIR}")
        print("❌ No valid video files found.")
        return
    else:
        print(f"✔ Found {len(video_files)} video(s) to process.")

    print(f"⚡ Found {len(video_files)} video(s). Extracting frames...")
    all_frames = process_videos_parallel(video_files, args.frame_rate, args.max_workers)

    print(f"📋 {len(all_frames)} frames extracted and saved.")
    print(f"✅ Frames saved at `{FRAME_DIR}` and logs at `{LOG_PATH}`.")

# -----------------------------#
# Run the script
# -----------------------------#
if __name__ == "__main__":
    main()
