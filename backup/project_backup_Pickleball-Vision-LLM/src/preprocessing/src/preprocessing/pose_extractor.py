import cv2
import mediapipe as mp
import numpy as np
import json
from typing import List, Dict, Any, Optional
import argparse

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseExtractor:
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize the PoseExtractor with MediaPipe Pose parameters.
        """
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_pose_from_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract pose landmarks from a single frame.
        :param frame: Input video frame as a NumPy array.
        :return: Dictionary containing pose landmarks and visibility scores, or None if no pose is detected.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            return {
                "landmarks": [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in results.pose_landmarks.landmark
                ]
            }
        return None

    def visualize_pose(self, frame: np.ndarray, landmarks: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Visualize pose landmarks on the frame.
        :param frame: Input video frame as a NumPy array.
        :param landmarks: Pose landmarks dictionary.
        :return: Frame with pose landmarks drawn.
        """
        if landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                mp_pose.PoseLandmarkList(
                    landmark=[
                        mp_pose.PoseLandmark(
                            x=lm["x"], y=lm["y"], z=lm["z"], visibility=lm["visibility"]
                        )
                        for lm in landmarks["landmarks"]
                    ]
                ),
                mp_pose.POSE_CONNECTIONS
            )
        return frame

    def process_video(self, video_path: str, save_visuals: bool = True, output_path: str = "output.mp4", frame_skip: int = 1) -> List[Dict]:
        """
        Process a video to extract pose landmarks for each frame.
        :param video_path: Path to the input video file.
        :param save_visuals: Whether to save the visualized output video.
        :param output_path: Path to save the visualized video.
        :param frame_skip: Number of frames to skip for faster processing.
        :return: List of dictionaries containing pose landmarks for each frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if save_visuals:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        poses = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                pose_data = self.extract_pose_from_frame(frame)
                if pose_data:
                    poses.append({"frame_idx": frame_idx, "pose": pose_data})

                if save_visuals and pose_data:
                    frame_with_pose = self.visualize_pose(frame, pose_data)
                    out.write(frame_with_pose)

            frame_idx += 1

        cap.release()
        if out:
            out.release()
        return poses

    def save_pose_json(self, poses: List[Dict], path: str):
        """
        Save pose landmarks to a JSON file.
        :param poses: List of pose landmarks dictionaries.
        :param path: Path to save the JSON file.
        """
        with open(path, 'w') as f:
            json.dump(poses, f, indent=4)
