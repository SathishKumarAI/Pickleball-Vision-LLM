import cv2
from typing import List, Dict
from prompt_templates import PromptTemplates

"""
video_llava_connector.py

Connector module for processing video frames and integrating with LLaVA for Pickleball Vision tasks.
Author: Sathish Kumar
Email: sathishkumar786.ml@gmail.com
"""



class VideoLLaVAConnector:
    """Handles video frame processing and LLaVA inference for Pickleball Vision tasks."""

    def __init__(self, llm_model, vision_model):
        """
        Initialize the connector with LLM and vision models.

        Args:
            llm_model: The language model for text-based tasks.
            vision_model: The vision model for frame analysis.
        """
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.prompts = PromptTemplates()

    def extract_frames(self, video_path: str, frame_rate: int = 1) -> List:
        """
        Extract frames from a video at a specified frame rate.

        Args:
            video_path: Path to the video file.
            frame_rate: Number of frames to extract per second.

        Returns:
            List of extracted frames as images.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, fps // frame_rate)

        success, frame = cap.read()
        count = 0
        while success:
            if count % frame_interval == 0:
                frames.append(frame)
            success, frame = cap.read()
            count += 1

        cap.release()
        return frames

    def analyze_frame(self, frame) -> str:
        """
        Analyze a single frame using the vision model and generate a caption.

        Args:
            frame: A single video frame.

        Returns:
            Caption describing the frame.
        """
        # Use the vision model to generate a description
        caption = self.vision_model.generate_caption(frame)
        return caption

    def process_video(self, video_path: str) -> List[Dict]:
        """
        Process a video and generate insights using LLaVA.

        Args:
            video_path: Path to the video file.

        Returns:
            List of insights for each frame or sequence.
        """
        frames = self.extract_frames(video_path)
        insights = []

        for frame in frames:
            # Analyze the frame
            caption = self.analyze_frame(frame)

            # Generate a coaching prompt
            coaching_prompt = self.prompts.get("clip_interpretation", caption=caption)

            # Use the LLM to generate coaching advice
            coaching_advice = self.llm_model.generate_response(coaching_prompt)

            insights.append({
                "frame_caption": caption,
                "coaching_advice": coaching_advice,
            })

        return insights


# Example usage
if __name__ == "__main__":
    # Mock models for demonstration
    class MockVisionModel:
        def generate_caption(self, frame):
            return "Player hits a forehand shot."

    class MockLLMModel:
        def generate_response(self, prompt):
            return "Focus on foot positioning and follow-through for better accuracy."

    vision_model = MockVisionModel()
    llm_model = MockLLMModel()

    connector = VideoLLaVAConnector(llm_model, vision_model)
    video_insights = connector.process_video("sample_video.mp4")

    for idx, insight in enumerate(video_insights):
        print(f"Frame {idx + 1}:")
        print("Caption:", insight["frame_caption"])
        print("Coaching Advice:", insight["coaching_advice"])
        print()