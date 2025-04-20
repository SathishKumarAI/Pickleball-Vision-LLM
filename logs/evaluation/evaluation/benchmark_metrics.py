from typing import List, Dict, Tuple, Any
import numpy as np
import time
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from datetime import datetime

"""
Benchmark metrics module for evaluating data filtering performance.
Tracks and analyzes various metrics related to frame filtering quality and efficiency.
"""


@dataclass
class FilterMetrics:
    """Data class to store filtering metrics for analysis."""
    total_frames: int = 0
    filtered_frames: int = 0
    processing_time: float = 0.0
    filter_reasons: Dict[str, int] = None
    avg_processing_time_per_frame: float = 0.0
    retention_rate: float = 0.0
    
    def __post_init__(self):
        self.filter_reasons = {
            "blurry": 0,
            "duplicate": 0,
            "static_scene": 0,
            "low_motion": 0,
            "outside_roi": 0,
            "missing_players": 0
        }

class BenchmarkMetrics:
    """Class for tracking and analyzing data filtering performance metrics."""
    
    def __init__(self, log_dir: str = "logs/filtering_metrics"):
        """
        Initialize the benchmark metrics tracker.
        
        Args:
            log_dir: Directory to store metric logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = FilterMetrics()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'filtering_metrics.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def start_session(self):
        """Start a new benchmarking session."""
        self.start_time = time.time()
        self.metrics = FilterMetrics()
        self.logger.info("Started new benchmarking session")

    def update_metrics(self, 
                      num_filtered: int,
                      filter_reasons: Dict[str, int],
                      batch_size: int):
        """
        Update metrics with latest batch results.
        
        Args:
            num_filtered: Number of filtered frames in this batch
            filter_reasons: Dictionary counting reasons for filtering
            batch_size: Size of the processed batch
        """
        self.metrics.total_frames += batch_size
        self.metrics.filtered_frames += num_filtered
        
        for reason, count in filter_reasons.items():
            self.metrics.filter_reasons[reason] += count

    def end_session(self) -> Dict[str, Any]:
        """
        End benchmarking session and compute final metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        self.metrics.processing_time = time.time() - self.start_time
        
        if self.metrics.total_frames > 0:
            self.metrics.avg_processing_time_per_frame = (
                self.metrics.processing_time / self.metrics.total_frames
            )
            self.metrics.retention_rate = (
                (self.metrics.total_frames - self.metrics.filtered_frames) 
                / self.metrics.total_frames
            )

        self._save_metrics()
        return self._get_metrics_dict()

    def _get_metrics_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "total_frames": self.metrics.total_frames,
            "filtered_frames": self.metrics.filtered_frames,
            "processing_time_seconds": round(self.metrics.processing_time, 2),
            "avg_processing_time_ms": round(self.metrics.avg_processing_time_per_frame * 1000, 2),
            "retention_rate": round(self.metrics.retention_rate, 3),
            "filter_reasons": self.metrics.filter_reasons,
            "timestamp": datetime.now().isoformat()
        }

    def _save_metrics(self):
        """Save metrics to JSON file."""
        metrics_dict = self._get_metrics_dict()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.log_dir / f"metrics_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
        self._log_summary(metrics_dict)

    def _log_summary(self, metrics: Dict[str, Any]):
        """Log summary of metrics."""
        self.logger.info(
            f"\nFiltering Summary:"
            f"\n- Total frames processed: {metrics['total_frames']}"
            f"\n- Frames filtered out: {metrics['filtered_frames']}"
            f"\n- Retention rate: {metrics['retention_rate']*100:.1f}%"
            f"\n- Average processing time: {metrics['avg_processing_time_ms']:.2f}ms/frame"
            f"\n- Filter reasons: {metrics['filter_reasons']}"
        )

def compute_efficiency_score(metrics: Dict[str, Any]) -> float:
    """
    Compute an efficiency score (0-1) based on metrics.
    
    Args:
        metrics: Dictionary of benchmark metrics
    
    Returns:
        Float representing overall efficiency score
    """
    # Weights for different components
    weights = {
        "speed": 0.3,  # Processing speed
        "retention": 0.3,  # Appropriate retention rate
        "distribution": 0.4  # Even distribution of filter reasons
    }
    
    # Speed score (lower is better)
    speed_score = min(1.0, 50 / metrics["avg_processing_time_ms"])
    
    # Retention score (aim for 0.3-0.7 retention rate)
    retention_rate = metrics["retention_rate"]
    retention_score = 1.0 - abs(0.5 - retention_rate)
    
    # Distribution score (how evenly distributed are the filter reasons)
    reasons = metrics["filter_reasons"]
    total_filtered = sum(reasons.values())
    if total_filtered == 0:
        distribution_score = 0
    else:
        proportions = [count/total_filtered for count in reasons.values() if count > 0]
        distribution_score = 1.0 - np.std(proportions)
    
    final_score = (
        weights["speed"] * speed_score +
        weights["retention"] * retention_score +
        weights["distribution"] * distribution_score
    )
    
    return round(final_score, 3)