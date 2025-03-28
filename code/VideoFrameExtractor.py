import os
import cv2
import json

class VideoFrameExtractor:
    def __init__(self, input_dir='/data/raw_videos/', output_dir='/data/frames/'):
        """
        Initialize frame extractor with input and output directories
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_frames(self, fps=5):
        """
        Extract frames from all videos in the input directory
        
        Args:
            fps (int): Frames per second to extract
        """
        for video_file in os.listdir(self.input_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(self.input_dir, video_file)
                video_id = os.path.splitext(video_file)[0]
                
                # Create video-specific output directory
                video_output_dir = os.path.join(self.output_dir, video_id)
                os.makedirs(video_output_dir, exist_ok=True)
                
                self._extract_frames_from_video(video_path, video_output_dir, fps)

    def _extract_frames_from_video(self, video_path, output_dir, fps):
        """
        Extract frames from a specific video
        
        Args:
            video_path (str): Path to the input video
            output_dir (str): Directory to save frames
            fps (int): Frames per second to extract
        """
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps // fps)
        
        frame_count = 0
        extracted_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
                cv2.imwrite(frame_filename, frame)
                extracted_frames.append({
                    'frame_path': frame_filename,
                    'frame_number': frame_count
                })
            
            frame_count += 1
        
        cap.release()
        
        # Optional: Save frame metadata
        metadata_path = os.path.join(output_dir, 'frame_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(extracted_frames, f, indent=2)

def main():
    extractor = VideoFrameExtractor()
    extractor.extract_frames(fps=5)
    print("Frame extraction completed!")

if __name__ == '__main__':
    main()