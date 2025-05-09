"""
Performance Metrics

This module provides functions for calculating and tracking
performance metrics for ball detection and tracking.
"""

import numpy as np
from typing import List, Dict, Tuple

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score (0-1)
    """
    # Calculate intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_detection_metrics(predictions: List[Dict[str, float]], 
                             ground_truth: List[Dict[str, float]], 
                             iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate detection performance metrics.
    
    Args:
        predictions: List of predicted detections
        ground_truth: List of ground truth annotations
        iou_threshold: Minimum IoU for true positive
        
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
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_tracking_metrics(predicted_trajectory: List[Dict[str, float]], 
                            ground_truth_trajectory: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate tracking performance metrics.
    
    Args:
        predicted_trajectory: List of predicted ball positions
        ground_truth_trajectory: List of ground truth positions
        
    Returns:
        Dictionary with tracking metrics (MOTA, MOTP)
    """
    if not ground_truth_trajectory or not predicted_trajectory:
        return {'mota': 0, 'motp': 0}
        
    # Calculate average position error
    errors = []
    for pred, gt in zip(predicted_trajectory, ground_truth_trajectory):
        error = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
        errors.append(error)
        
    motp = np.mean(errors) if errors else float('inf')
    
    # Simple MOTA calculation (can be expanded)
    mota = 1.0 - (motp / 100)  # Normalize to 0-1 range
    mota = max(0, min(1, mota))  # Clip to valid range
    
    return {
        'mota': mota,  # Multi-Object Tracking Accuracy
        'motp': motp   # Multi-Object Tracking Precision
    } 