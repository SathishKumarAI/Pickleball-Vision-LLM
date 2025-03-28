import os
import csv
import yt_dlp
from datetime import datetime

class PaddlePickleballDownloader:
    def __init__(self, output_dir='/data/raw_videos/'):
        """
        Initialize the downloader with a specific output directory
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metadata tracking
        self.metadata_file = os.path.join(output_dir, 'pickleball_matches_metadata.csv')
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(output_dir, '%(id)s_%(title)s.%(ext)s'),
            'nooverwrites': True,
            'no_color': True,
            'no_warnings': True,
            'ignoreerrors': False,
            'restrict_filenames': True,
        }

    def download_videos(self, video_urls):
        """
        Download YouTube videos and log metadata
        """
        downloaded_videos = []
        
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            for url in video_urls:
                try:
                    info_dict = ydl.extract_info(url, download=True)
                    video_id = info_dict.get('id', None)
                    title = info_dict.get('title', 'Unknown Title')
                    duration = info_dict.get('duration', 0)
                    upload_date = info_dict.get('upload_date', 'Unknown')
                    
                    video_info = {
                        'video_id': video_id,
                        'title': title,
                        'url': url,
                        'duration_seconds': duration,
                        'upload_date': upload_date,
                        'download_timestamp': datetime.now().isoformat()
                    }
                    downloaded_videos.append(video_info)
                    
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
        
        self._log_metadata(downloaded_videos)
        return downloaded_videos

    def _log_metadata(self, video_metadata):
        """
        Log video metadata to a CSV file
        """
        file_exists = os.path.exists(self.metadata_file)
        
        with open(self.metadata_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'video_id', 'title', 'url', 
                'duration_seconds', 'upload_date', 
                'download_timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for video in video_metadata:
                writer.writerow(video)

def main():
    # Pickleball match video URLs (replace with actual URLs)
    pickleball_urls = [
        'https://www.youtube.com/watch?v=example1',
        'https://www.youtube.com/watch?v=example2',
        'https://www.youtube.com/watch?v=example3',
        'https://www.youtube.com/watch?v=example4',
        'https://www.youtube.com/watch?v=example5',
        'https://www.youtube.com/watch?v=example6'
    ]

    downloader = PaddlePickleballDownloader()
    downloaded_videos = downloader.download_videos(pickleball_urls)
    
    print(f"Successfully downloaded {len(downloaded_videos)} videos")

if __name__ == '__main__':
    main()