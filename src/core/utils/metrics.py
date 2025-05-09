import numpy as np
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