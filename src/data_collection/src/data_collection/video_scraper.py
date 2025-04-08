from pathlib import Path
from datetime import datetime
from yt_dlp import YoutubeDL
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------#
# Config
# -----------------------------#
VIDEO_DIR = Path("data/raw_videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
CSV_LOG_PATH = Path("data/video_metadata.csv")
MAX_WORKERS = 3  # Tune based on bandwidth/CPU

# List of Pickleball YouTube video URLs
video_urls = [
    "https://www.youtube.com/watch?v=jGj5AkuaDDY",

]

# -----------------------------#
# Extract Metadata Function
# -----------------------------#
def fetch_video_metadata(url):
    ydl_opts = {
        "quiet": True,
        "skip_download": True
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title"),
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
                "url": url,
                "filename": VIDEO_DIR / f"{info.get('title')}.mp4",
                "downloaded_at": None
            }
        except Exception as e:
            return {"url": url, "status": "metadata_failed", "error": str(e)}

# -----------------------------#
# Download Video Function
# -----------------------------#
def download_video(meta):
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": str(meta["filename"]),
        "quiet": True,
        "merge_output_format": "mp4"
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([meta["url"]])
            meta["downloaded_at"] = datetime.now().isoformat()
            meta["status"] = "success"
        except Exception as e:
            meta["status"] = "failed"
            meta["error"] = str(e)
    return meta

# -----------------------------#
# Main
# -----------------------------#
def main():
    metadata = []

    print("🔍 Fetching video metadata...")
    for url in tqdm(video_urls):
        meta = fetch_video_metadata(url)
        metadata.append(meta)

    print("⚡ Downloading videos concurrently...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_meta = {executor.submit(download_video, meta): meta for meta in metadata if "filename" in meta}
        for future in tqdm(as_completed(future_to_meta), total=len(future_to_meta)):
            results.append(future.result())

    # Save metadata to CSV
    df = pd.DataFrame(results)
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_LOG_PATH, index=False)
    print(f"\n✅ All downloads complete. Metadata saved to {CSV_LOG_PATH}")

if __name__ == "__main__":
    main()
