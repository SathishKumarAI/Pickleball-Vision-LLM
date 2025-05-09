"""Metrics collection and monitoring system."""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

@dataclass
class MetricPoint:
    """Single metric measurement."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class MetricSeries:
    """Time series of metric measurements."""
    name: str
    points: List[MetricPoint] = field(default_factory=list)
    
    def add_point(self, value: float, tags: Optional[Dict[str, str]] = None):
        """Add a new measurement."""
        self.points.append(MetricPoint(self.name, value, time.time(), tags or {}))
        
    @property
    def values(self) -> List[float]:
        """Get list of values."""
        return [p.value for p in self.points]
        
    @property
    def timestamps(self) -> List[float]:
        """Get list of timestamps."""
        return [p.timestamp for p in self.points]

class MetricsCollector:
    """Collect and store metrics."""
    
    def __init__(self, output_dir: str):
        """Initialize collector.
        
        Args:
            output_dir: Directory to save metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, MetricSeries] = {}
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric measurement.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional key-value tags
        """
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name)
        self.metrics[name].add_point(value, tags)
        
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get metric series by name."""
        return self.metrics.get(name)
        
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for metric."""
        series = self.get_metric(name)
        if series and series.points:
            return series.points[-1].value
        return None
        
    def save_metrics(self, filename: Optional[str] = None):
        """Save metrics to JSON file.
        
        Args:
            filename: Output filename, defaults to timestamp
        """
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        data = {
            name: {
                'values': series.values,
                'timestamps': series.timestamps,
                'points': [
                    {
                        'value': p.value,
                        'timestamp': p.timestamp,
                        'tags': p.tags
                    }
                    for p in series.points
                ]
            }
            for name, series in self.metrics.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_metrics(self, filename: str):
        """Load metrics from JSON file.
        
        Args:
            filename: Input filename
        """
        input_path = self.output_dir / filename
        
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        for name, series_data in data.items():
            series = MetricSeries(name)
            for point in series_data['points']:
                series.add_point(point['value'], point.get('tags', {}))
            self.metrics[name] = series 