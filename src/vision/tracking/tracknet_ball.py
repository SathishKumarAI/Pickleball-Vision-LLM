"""TrackNet ball tracker adapter — heatmap-based, for the fast/small pickleball.

IoU/nearest tracking loses a fast ball; TrackNet predicts a per-frame heatmap from
a sliding window of frames and takes its peak as the ball position. We wrap a
pretrained TrackNet (vendored from yastrebksv/TrackNet or qaz812345/TrackNetV3 on
the GPU box) behind a small interface that yields per-frame centroids the
trajectory analyzer consumes.

Lazy-imports torch/cv2 — parse-clean here; runs on the GPU box. The actual model
load is intentionally a single seam (`_load_model`) to vendor the chosen repo.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

Centroid = Optional[Tuple[float, float]]


class TrackNetBall:
    """Predict ball centroids across a frame sequence via TrackNet heatmaps."""

    def __init__(self, weights: Optional[str] = None, window: int = 3,
                 input_size: Tuple[int, int] = (360, 640), device: str = "cuda"):
        self.weights = weights or os.getenv("TRACKNET_WEIGHTS", "")
        self.window = window
        self.input_size = input_size
        self.device = device
        self._model = None

    def _load_model(self):
        # Vendor the TrackNet repo + weights on the GPU box; load here.
        # (qaz812345/TrackNetV3 or yastrebksv/TrackNet). Kept as one seam.
        if self._model is None:
            import torch  # lazy
            if not self.weights or not os.path.exists(self.weights):
                raise RuntimeError("TrackNet weights not found; set TRACKNET_WEIGHTS")
            self._model = torch.load(self.weights, map_location=self.device)
            self._model.eval()
        return self._model

    def predict(self, frames: Sequence) -> List[Centroid]:
        """Return a per-frame ball centroid (image px) for the given BGR frames."""
        import cv2  # lazy
        import numpy as np
        import torch  # lazy

        model = self._load_model()
        h, w = self.input_size
        centroids: List[Centroid] = [None] * len(frames)
        for i in range(self.window - 1, len(frames)):
            stack = []
            for j in range(i - self.window + 1, i + 1):
                f = cv2.resize(frames[j], (w, h))
                stack.append(f[..., ::-1] / 255.0)  # BGR->RGB, normalize
            inp = torch.tensor(np.concatenate(stack, axis=2).transpose(2, 0, 1)[None],
                               dtype=torch.float32, device=self.device)
            with torch.no_grad():
                heat = model(inp)[0, 0].cpu().numpy()
            if heat.max() > 0.5:
                y, x = np.unravel_index(int(heat.argmax()), heat.shape)
                # scale heatmap coords back to original frame size
                fh, fw = frames[i].shape[:2]
                centroids[i] = (x * fw / heat.shape[1], y * fh / heat.shape[0])
        return centroids
