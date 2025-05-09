# !pip install ultralytics opencv-python pandas

from ultralytics import YOLO
import cv2
import pandas as pd
import time
import numpy as np
import torch
import os

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create output directory for frames
os.makedirs('output_frames', exist_ok=True)
os.makedirs('output_masks', exist_ok=True)

# Load segmentation model
model = YOLO('yolov8x-seg.pt')  # Using YOLOv8x for better segmentation
model.to(device)

# Load video
video_path = r"data/raw_videos/Johns_Tardio v McGuffin_Duong at the CIBC Texas Open.mp4"
cap = cv2.VideoCapture(video_path)

# Output setup
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate frame skip to get optimal frames
FRAME_SKIP = 30  # Process every 30th frame (1 frame per second at 30fps)
MAX_FRAMES = 100  # Process maximum 100 frames
total_frames_to_process = min(MAX_FRAMES, total_frames // FRAME_SKIP)

print(f"Video info:")
print(f"- Total frames: {total_frames}")
print(f"- FPS: {fps}")
print(f"- Processing every {FRAME_SKIP}th frame")
print(f"- Will process {total_frames_to_process} frames")

out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Define colors for different classes
COLORS = {
    'person': (0, 255, 0),    # Green for players
    'sports ball': (0, 0, 255),  # Red for ball
    'chair': (255, 0, 0),     # Blue for audience
    'bench': (255, 0, 0),     # Blue for audience
}

# Function to create colored mask
def create_colored_mask(mask, color):
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask > 0] = color
    return colored_mask

detection_log = []
frame_count = 0
processed_count = 0
start_time = time.time()

print("\nStarting video processing...")
print("Performing instance segmentation...")
print(f"Processing on {device}")

while cap.isOpened() and processed_count < total_frames_to_process:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames
    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue

    # Inference with GPU acceleration
    results = model(frame, verbose=False, device=device)
    
    # Create overlay for masks
    overlay = frame.copy()
    
    # Process detections
    for result in results:
        boxes = result.boxes
        masks = result.masks
        
        if masks is not None:
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                
                # Only process relevant classes
                if class_name in ['person', 'sports ball', 'chair', 'bench']:
                    # Get mask
                    mask_np = mask.data[0].cpu().numpy()
                    
                    # Create colored mask
                    color = COLORS.get(class_name, (255, 255, 255))
                    colored_mask = create_colored_mask(mask_np, color)
                    
                    # Apply mask to overlay
                    mask_area = mask_np > 0
                    overlay[mask_area] = cv2.addWeighted(overlay[mask_area], 0.7, 
                                                       colored_mask[mask_area], 0.3, 0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Log detection
                    detection_log.append({
                        "frame": frame_count,
                        "class": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "mask_area": int(np.sum(mask_np))
                    })

    # Blend original frame with overlay
    output = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Add frame counter and FPS
    elapsed_time = time.time() - start_time
    fps_current = processed_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(output, f"Frame: {frame_count} FPS: {fps_current:.1f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Write frame
    out.write(output)
    
    # Save first 5 processed frames and their masks
    if processed_count < 5:
        # Save original frame with detections
        frame_path = f'output_frames/frame_{frame_count:03d}.jpg'
        cv2.imwrite(frame_path, output)
        
        # Save mask visualization
        mask_path = f'output_masks/mask_{frame_count:03d}.jpg'
        cv2.imwrite(mask_path, overlay)
        
        print(f"\nFrame {frame_count} saved to {frame_path}")
        print("Detections in this frame:")
        frame_detections = [d for d in detection_log if d['frame'] == frame_count]
        for det in frame_detections:
            print(f"- {det['class']} (confidence: {det['confidence']:.2f}, area: {det['mask_area']} pixels)")
    
    frame_count += 1
    processed_count += 1

    # Print progress
    if processed_count % 10 == 0:
        print(f"Processed {processed_count}/{total_frames_to_process} frames... FPS: {fps_current:.1f}")

cap.release()
out.release()

# Save detections to CSV
df = pd.DataFrame(detection_log)
df.to_csv("detections.csv", index=False)

print("\nProcessing complete!")
print(f"Total frames processed: {processed_count}")
print(f"Average FPS: {processed_count/elapsed_time:.1f}")
print(f"Detections saved to: detections.csv")
print(f"Annotated video saved as: output.mp4")
print(f"First 5 frames and masks saved in output_frames and output_masks directories")

# Show summary of detections
print("\nDetection Summary:")
print(df['class'].value_counts())

# Show average mask area per class
print("\nAverage mask area per class (pixels):")
print(df.groupby('class')['mask_area'].mean().round(2))
