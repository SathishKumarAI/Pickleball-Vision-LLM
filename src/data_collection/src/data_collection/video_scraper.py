import os
import re
import csv
import time
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from yt_dlp import YoutubeDL

# -----------------------------#
# Configuration
# -----------------------------#
VIDEO_DIR = Path("data/raw_videos")
CSV_LOG_PATH = Path("data/video_metadata.csv")
LOG_PATH = Path("logs/video_downloader.log")

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = int(os.getenv("MAX_WORKERS", 3))

VIDEO_URLS = [
    "https://www.youtube.com/watch?v=jGj5AkuaDDY",
]

# -----------------------------#
# Logging Setup
# -----------------------------#
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------#
# Utilities
# -----------------------------#
def is_valid_url(url):
    """Check if a URL is a valid YouTube link."""
    return re.match(r"(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+", url) is not None

def sanitize_filename(filename):
    """Remove illegal characters for filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

# -----------------------------#
# Metadata Extraction
# -----------------------------#
def fetch_video_metadata(url):
    """Fetch metadata for a given YouTube video."""
    try:
        with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = sanitize_filename(info.get("title"))
            return {
                "title": info.get("title"),
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
                "url": url,
                "filename": VIDEO_DIR / f"{title}.mp4",
                "downloaded_at": None,
                "status": None,
                "error": None
            }
    except Exception as e:
        logging.error(f"Metadata fetch failed for {url}: {e}")
        return {
            "url": url,
            "status": "metadata_failed",
            "error": str(e)
        }

# -----------------------------#
# Video Downloading
# -----------------------------#
def download_video(meta, retries=2):
    """Download a video from YouTube using yt-dlp."""
    if meta.get("filename") and Path(meta["filename"]).exists():
        logging.info(f"Already downloaded: {meta['filename'].name}")
        meta["status"] = "skipped"
        return meta

    for attempt in range(retries + 1):
        try:
            with YoutubeDL({
                "format": "bestvideo+bestaudio/best",
                "outtmpl": str(meta["filename"]),
                "quiet": True,
                "merge_output_format": "mp4"
            }) as ydl:
                ydl.download([meta["url"]])
                meta["downloaded_at"] = datetime.now().isoformat()
                meta["status"] = "success"
                return meta
        except Exception as e:
            logging.error(f"Download attempt {attempt+1} failed for {meta['url']}: {e}")
            meta["status"] = "failed"
            meta["error"] = str(e)
            time.sleep(2)
    return meta

# -----------------------------#
# Main Execution Flow
# -----------------------------#
def main():
    print("🔍 Validating and fetching metadata...")
    valid_urls = [url for url in VIDEO_URLS if is_valid_url(url)]
    if not valid_urls:
        print("❌ No valid YouTube URLs found.")
        return

    metadata = [fetch_video_metadata(url) for url in tqdm(valid_urls, desc="Metadata")]

    print("⚡ Starting downloads...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_video, meta): meta for meta in metadata if meta.get("status") != "metadata_failed"
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloads"):
            results.append(future.result())

    print("📋 Saving metadata to CSV...")
    df = pd.DataFrame(results)
    df.to_csv(CSV_LOG_PATH, index=False)

    print(f"✅ Done. Metadata saved at `{CSV_LOG_PATH}` and logs at `{LOG_PATH}`.")

# -----------------------------#
# Entry Point
# -----------------------------#
if __name__ == "__main__":
    main()
