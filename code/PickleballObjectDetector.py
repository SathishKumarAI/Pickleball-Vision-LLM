import os
import json
import cv2
from ultralytics import YOLO

class PickleballObjectDetector:
    def __init__(self, 
                 frames_dir='/data/frames/', 
                 output_dir='/data/detections/',
                 model_path='yolov8n.pt'):
        """
        Initialize object detector with directories and YOLO model
        
        Args:
            frames_dir (str): Directory containing video frames
            output_dir (str): Directory to save detection results
            model_path (str): Path to YOLO model weights
        """
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load YOLO model
        self.model = YOLO(model_path)

    def detect_objects(self, confidence_threshold=0.5):
        """
        Run object detection on frames
        
        Args:
            confidence_threshold (float): Minimum confidence to report detection
        """
        for video_folder in os.listdir(self.frames_dir):
            video_path = os.path.join(self.frames_dir, video_folder)
            
            if not os.path.isdir(video_path):
                continue
            
            detection_output_dir = os.path.join(self.output_dir, video_folder)
            os.makedirs(detection_output_dir, exist_ok=True)
            
            self._process_video_frames(video_path, detection_output_dir, confidence_threshold)

    def _process_video_frames(self, frame_dir, output_dir, confidence_threshold):
        """
        Process frames for a single video
        
        Args:
            frame_dir (str): Directory of video frames
            output_dir (str): Directory to save detection results
            confidence_threshold (float): Minimum confidence for detection
        """
        all_detections = []
        
        for frame_file in sorted(os.listdir(frame_dir)):
            if frame_file.endswith(('.jpg', '.png')):
                frame_path = os.path.join(frame_dir, frame_file)
                results = self.model(frame_path, conf=confidence_threshold)
                
                frame_detections = []
                for result in results:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # Extract detection details
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection = {
                            'class': self.model.names[cls],
                            'confidence': conf,
                            'bbox': {
                                'x1': x1, 'y1': y1,
                                'x2': x2, 'y2': y2
                            }
                        }
                        frame_detections.append(detection)
                
                # Visualize and save detected frame
                frame = cv2.imread(frame_path)
                for det in frame_detections:
                    x1, y1 = int(det['bbox']['x1']), int(det['bbox']['y1'])
                    x2, y2 = int(det['bbox']['x2']), int(det['bbox']['y2'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 
                                f"{det['class']} {det['confidence']:.2f}", 
                                (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                (0, 255, 0), 2)
                
                output_frame_path = os.path.join(output_dir, f'detected_{frame_file}')
                cv2.imwrite(output_frame_path, frame)
                
                # Log detections
                all_detections.append({
                    'frame': frame_file,
                    'detections': frame_detections
                })
        
        # Save detection metadata
        metadata_path = os.path.join(output_dir, 'detections_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(all_detections, f, indent=2)

def main():
    detector = PickleballObjectDetector()
    detector.detect_objects(confidence_threshold=0.5)
    print("Object detection completed!")

if __name__ == '__main__':
    main()