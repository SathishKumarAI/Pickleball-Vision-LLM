# Ball Detection Prompts

This document contains prompts for ball detection and tracking tasks.

## Initial Ball Detection

```prompt
Detect pickleball in the given frame:
- Identify circular objects matching ball dimensions
- Apply confidence thresholds (minimum: 0.75)
- Return bounding box coordinates
- Include confidence score

Input: [preprocessed frame]
Expected Output: {
    "bbox": [x1, y1, x2, y2],
    "confidence": float,
    "frame_id": int
}
```

## Ball Tracking

```prompt
Track ball movement across sequential frames:
- Maintain ball trajectory
- Handle occlusions
- Predict next position
- Filter false positives

Input: [sequence of frames with detections]
Expected Output: {
    "trajectory": [[x, y, frame_id], ...],
    "predictions": [[x, y], ...],
    "confidence": float
}
```

## Multiple Ball Handling

```prompt
Handle scenarios with multiple balls:
- Distinguish between active and inactive balls
- Track primary ball of interest
- Handle ball switches
- Maintain ball IDs

Input: [frame with multiple detections]
Expected Output: {
    "balls": [
        {"id": int, "bbox": [x1, y1, x2, y2], "active": bool},
        ...
    ]
}
```

## Usage Instructions

1. Select appropriate prompt based on detection scenario
2. Ensure input frames are preprocessed
3. Apply detection model with given parameters
4. Validate detection outputs
5. Log unusual detection patterns

## Best Practices

- Run quality checks before detection
- Maintain detection history
- Log false positives for model improvement
- Use temporal consistency checks
- Document detection failures 