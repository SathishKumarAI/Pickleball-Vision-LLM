import numpy as np
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

"""
metrics.py

Utility functions for evaluating model performance in the Pickleball Vision Model project.
Includes metrics for tracking, classification, and game state prediction.

Author: GitHub Copilot
"""


def compute_classification_metrics(y_true, y_pred, average='weighted'):
    """
    Compute precision, recall, F1-score, and accuracy for classification tasks.

    Args:
        y_true (list or np.ndarray): Ground truth labels.
        y_pred (list or np.ndarray): Predicted labels.
        average (str): Averaging method for multi-class metrics. Default is 'weighted'.

    Returns:
        dict: Dictionary containing precision, recall, F1-score, and accuracy.
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy
    }

def compute_tracking_metrics(gt_tracks, pred_tracks, iou_threshold=0.5):
    """
    Compute tracking metrics such as MOTA (Multiple Object Tracking Accuracy) and MOTP (Multiple Object Tracking Precision).

    Args:
        gt_tracks (list): List of ground truth bounding boxes for each frame.
        pred_tracks (list): List of predicted bounding boxes for each frame.
        iou_threshold (float): IoU threshold to consider a match. Default is 0.5.

    Returns:
        dict: Dictionary containing MOTA and MOTP metrics.
    """
    # Placeholder implementation for IoU and tracking metrics
    # Replace with actual implementation based on project requirements
    mota = 0.0  # Multiple Object Tracking Accuracy
    motp = 0.0  # Multiple Object Tracking Precision

    # TODO: Implement IoU calculation and tracking metrics
    return {
        "MOTA": mota,
        "MOTP": motp
    }

def compute_pose_estimation_metrics(gt_poses, pred_poses, threshold=0.1):
    """
    Compute metrics for pose estimation, such as Percentage of Correct Keypoints (PCK).

    Args:
        gt_poses (np.ndarray): Ground truth keypoints (N x K x 2).
        pred_poses (np.ndarray): Predicted keypoints (N x K x 2).
        threshold (float): Distance threshold for considering a keypoint correct.

    Returns:
        dict: Dictionary containing PCK metric.
    """
    num_keypoints = gt_poses.shape[1]
    correct_keypoints = 0
    total_keypoints = 0

    for gt, pred in zip(gt_poses, pred_poses):
        distances = np.linalg.norm(gt - pred, axis=1)
        correct_keypoints += np.sum(distances < threshold)
        total_keypoints += num_keypoints

    pck = correct_keypoints / total_keypoints if total_keypoints > 0 else 0.0

    return {
        "PCK": pck
    }

def compute_game_state_metrics(gt_states, pred_states):
    """
    Compute accuracy for game state prediction.

    Args:
        gt_states (list): Ground truth game states.
        pred_states (list): Predicted game states.

    Returns:
        dict: Dictionary containing accuracy metric.
    """
    accuracy = accuracy_score(gt_states, pred_states)
    return {
        "accuracy": accuracy
    }


# ---------------------------------------------------------------------------
# Detection / tracking metrics (merged from the former src/fusion/utils/metrics.py).
# These provide the concrete IoU + detection/tracking implementation that the
# compute_tracking_metrics() stub above left as a TODO.
#
# TODO(oss): replace the hand-rolled calculate_iou with torchvision.ops.box_iou
# or supervision's metrics once those deps are installed — don't reinvent.
# ---------------------------------------------------------------------------

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU score (0-1)
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_detection_metrics(predictions: List[Dict[str, float]],
                                ground_truth: List[Dict[str, float]],
                                iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate detection performance metrics (precision/recall/F1) by IoU match.

    Args:
        predictions: List of predicted detections (each with a 'bbox' key)
        ground_truth: List of ground truth annotations (each with a 'bbox' key)
        iou_threshold: Minimum IoU for a true positive

    Returns:
        Dictionary with precision, recall, and F1 score
    """
    if not ground_truth:
        return {'precision': 0, 'recall': 0, 'f1': 0}

    true_positives = 0
    false_positives = len(predictions)

    for gt in ground_truth:
        for pred in predictions:
            if calculate_iou(gt['bbox'], pred['bbox']) >= iou_threshold:
                true_positives += 1
                false_positives -= 1
                break

    precision = true_positives / len(predictions) if predictions else 0
    recall = true_positives / len(ground_truth)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def calculate_tracking_metrics(predicted_trajectory: List[Dict[str, float]],
                               ground_truth_trajectory: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate tracking performance metrics (MOTA / MOTP) from trajectories.

    Args:
        predicted_trajectory: List of predicted ball positions (each with 'x','y')
        ground_truth_trajectory: List of ground truth positions (each with 'x','y')

    Returns:
        Dictionary with MOTA and MOTP
    """
    if not ground_truth_trajectory or not predicted_trajectory:
        return {'mota': 0, 'motp': 0}

    errors = []
    for pred, gt in zip(predicted_trajectory, ground_truth_trajectory):
        error = np.sqrt((pred['x'] - gt['x']) ** 2 + (pred['y'] - gt['y']) ** 2)
        errors.append(error)

    motp = np.mean(errors) if errors else float('inf')
    mota = max(0, min(1, 1.0 - (motp / 100)))

    return {'mota': mota, 'motp': motp}