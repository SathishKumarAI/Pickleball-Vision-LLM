import cv2
import numpy as np
import random
from typing import List, Tuple

def flip_frame(frame: np.ndarray, flip_code: int = 1) -> np.ndarray:
    """
    Flip the frame horizontally or vertically.

    Args:
        frame (np.ndarray): Input video frame.
        flip_code (int): Flip code (0 for vertical, 1 for horizontal, -1 for both).

    Returns:
        np.ndarray: Flipped frame.
    """
    return cv2.flip(frame, flip_code)


def rotate_frame(frame: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the frame by a given angle.

    Args:
        frame (np.ndarray): Input video frame.
        angle (float): Angle to rotate the frame.

    Returns:
        np.ndarray: Rotated frame.
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, rotation_matrix, (w, h))


def adjust_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust the brightness of the frame.

    Args:
        frame (np.ndarray): Input video frame.
        factor (float): Brightness adjustment factor (>1 for brighter, <1 for darker).

    Returns:
        np.ndarray: Brightness-adjusted frame.
    """
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)


def add_noise(frame: np.ndarray, noise_type: str = "gaussian") -> np.ndarray:
    """
    Add noise to the frame.

    Args:
        frame (np.ndarray): Input video frame.
        noise_type (str): Type of noise to add ('gaussian' or 'salt_pepper').

    Returns:
        np.ndarray: Frame with added noise.
    """
    if noise_type == "gaussian":
        mean = 0
        stddev = 15
        noise = np.random.normal(mean, stddev, frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(frame, noise)
    elif noise_type == "salt_pepper":
        noisy_frame = frame.copy()
        prob = 0.02
        num_salt = int(prob * frame.size * 0.5)
        num_pepper = int(prob * frame.size * 0.5)

        # Add salt
        coords = [np.random.randint(0, i - 1, num_salt) for i in frame.shape[:2]]
        noisy_frame[coords[0], coords[1]] = 255

        # Add pepper
        coords = [np.random.randint(0, i - 1, num_pepper) for i in frame.shape[:2]]
        noisy_frame[coords[0], coords[1]] = 0
    else:
        raise ValueError("Unsupported noise type. Use 'gaussian' or 'salt_pepper'.")
    return noisy_frame


def augment_frame(frame: np.ndarray) -> List[np.ndarray]:
    """
    Apply a series of augmentations to a frame.

    Args:
        frame (np.ndarray): Input video frame.

    Returns:
        List[np.ndarray]: List of augmented frames.
    """
    augmented_frames = []

    # Flip
    augmented_frames.append(flip_frame(frame, flip_code=1))  # Horizontal flip
    augmented_frames.append(flip_frame(frame, flip_code=0))  # Vertical flip

    # Rotate
    for angle in [-15, 15]:
        augmented_frames.append(rotate_frame(frame, angle))

    # Brightness adjustment
    augmented_frames.append(adjust_brightness(frame, factor=1.2))  # Brighter
    augmented_frames.append(adjust_brightness(frame, factor=0.8))  # Darker

    # Add noise
    augmented_frames.append(add_noise(frame, noise_type="gaussian"))
    augmented_frames.append(add_noise(frame, noise_type="salt_pepper"))

    return augmented_frames


def augment_video_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply augmentations to a list of video frames.

    Args:
        frames (List[np.ndarray]): List of video frames.

    Returns:
        List[np.ndarray]: List of augmented frames.
    """
    augmented_frames = []
    for frame in frames:
        augmented_frames.extend(augment_frame(frame))
    return augmented_frames


if __name__ == "__main__":
    # Example usage
    sample_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Dummy black frame
    augmented = augment_frame(sample_frame)

    for idx, aug_frame in enumerate(augmented):
        cv2.imshow(f"Augmented Frame {idx}", aug_frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()