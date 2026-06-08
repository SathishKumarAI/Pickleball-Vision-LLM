"""Vision intelligence — trajectory analysis, shot classification, rally segmentation.

Pure-NumPy analytics over the structured outputs of the vision pipeline (no torch/
cv2/GPU), so they run and unit-test anywhere. Heavier model adapters live alongside
under ``src/vision/`` and lazy-import their deps.
"""
