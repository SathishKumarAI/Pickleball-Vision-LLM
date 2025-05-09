import os
import logging
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SegmentationSAM:
    """
    A class for segmenting objects in pickleball gameplay videos using the Segment Anything Model (SAM).
    """

    def __init__(self, model_type: str = "vit_h", checkpoint_path: str = "sam_vit_h_4b8939.pth", device: str = "cuda"):
        """
        Initialize the SAM model.

        Args:
            model_type (str): The type of SAM model to use (e.g., "vit_h", "vit_l").
            checkpoint_path (str): Path to the SAM model checkpoint.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path

        # Load the SAM model
        logger.info("Loading SAM model...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        self.predictor = SamPredictor(self.sam)
        logger.info("SAM model loaded successfully.")

    def segment_frame(self, frame: torch.Tensor, input_points: list, input_labels: list):
        """
        Perform segmentation on a single frame.

        Args:
            frame (torch.Tensor): The input frame as a PyTorch tensor.
            input_points (list): List of points for segmentation.
            input_labels (list): Labels corresponding to the input points.

        Returns:
            masks (torch.Tensor): Segmentation masks for the frame.
        """
        try:
            logger.info("Setting image for SAM predictor...")
            self.predictor.set_image(frame)

            logger.info("Predicting segmentation masks...")
            masks, _, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            logger.info("Segmentation completed.")
            return masks
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            raise

    @staticmethod
    def preprocess_frame(frame: np.ndarray):
        """
        Preprocess a frame for SAM input.

        Args:
            frame (np.ndarray): The input frame as a NumPy array.

        Returns:
            torch.Tensor: The preprocessed frame as a PyTorch tensor.
        """
        logger.info("Preprocessing frame for SAM...")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        return frame_tensor

    @staticmethod
    def visualize_masks(frame: np.ndarray, masks: torch.Tensor):
        """
        Visualize segmentation masks on the frame.

        Args:
            frame (np.ndarray): The original frame.
            masks (torch.Tensor): Segmentation masks.

        Returns:
            np.ndarray: Frame with masks overlaid.
        """
        logger.info("Visualizing segmentation masks...")
        overlay = frame.copy()
        for mask in masks:
            color = (0, 255, 0)  # Green color for masks
            overlay[mask > 0] = color
        return overlay


def main():
    # Example usage
    video_path = "path_to_video.mp4"
    checkpoint_path = "path_to_sam_checkpoint.pth"

    # Initialize the segmentation module
    segmenter = SegmentationSAM(checkpoint_path=checkpoint_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_tensor = segmenter.preprocess_frame(frame)

        # Example input points and labels (to be replaced with actual logic)
        input_points = [[100, 200], [300, 400]]  # Example points
        input_labels = [1, 0]  # Example labels (1 for foreground, 0 for background)

        # Perform segmentation
        masks = segmenter.segment_frame(frame_tensor, input_points, input_labels)

        # Visualize masks
        overlay = segmenter.visualize_masks(frame, masks)

        # Display the frame with masks
        cv2.imshow("Segmentation", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()