# Metrics Monitoring

The metrics monitoring system provides tools for collecting, storing, and analyzing performance metrics.

## Overview

The system consists of:
- `MetricPoint`: Single measurement with timestamp
- `MetricSeries`: Time series of measurements
- `MetricsCollector`: Main collection and storage interface

## Usage

```python
from pickleball_vision.monitoring.metrics import MetricsCollector

# Initialize collector
collector = MetricsCollector("output/metrics")

# Record metrics
collector.record_metric("confidence", 0.95, tags={"model": "v1"})
collector.record_metric("quality", 0.85)

# Get latest values
latest_confidence = collector.get_latest("confidence")

# Save metrics to file
collector.save_metrics("results.json")

# Load metrics from file
collector.load_metrics("results.json")
```

## Metric Points

Each `MetricPoint` contains:
- Name: Metric identifier
- Value: Numerical measurement
- Timestamp: Automatically added
- Tags: Optional key-value metadata

## Metric Series

The `MetricSeries` class:
- Maintains ordered list of points
- Provides value and timestamp accessors
- Supports adding new measurements
- Preserves measurement history

## Metrics Collector

The collector provides:

### Recording
```python
collector.record_metric(name, value, tags)
```
- Creates series if not exists
- Adds point with current timestamp
- Supports optional tagging

### Retrieval
```python
series = collector.get_metric(name)
latest = collector.get_latest(name)
```
- Get full series or latest value
- Returns None if metric not found

### Storage
```python
collector.save_metrics(filename)
collector.load_metrics(filename)
```
- JSON file format
- Preserves all metadata
- Supports timestamp-based filenames 