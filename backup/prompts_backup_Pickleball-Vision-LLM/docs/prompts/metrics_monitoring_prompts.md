# Metrics and Monitoring Prompts

This document contains prompts for metrics collection and monitoring tasks.

## Performance Metrics Collection

```prompt
Collect and analyze system performance metrics:
- Track detection confidence scores
- Measure processing time per frame
- Monitor memory usage
- Calculate detection rate

Input: [system metrics data]
Expected Output: {
    "confidence_avg": float,
    "processing_time_ms": float,
    "memory_usage_mb": float,
    "detection_rate": float
}
```

## Quality Metrics Analysis

```prompt
Analyze quality metrics across video segments:
- Frame quality distribution
- Detection consistency
- False positive rates
- System reliability

Input: [quality metrics data]
Expected Output: {
    "quality_scores": [float, ...],
    "detection_consistency": float,
    "false_positive_rate": float,
    "system_reliability": float
}
```

## Alert Generation

```prompt
Generate alerts based on metric thresholds:
- Detection confidence drops
- Processing delays
- Resource constraints
- System errors

Input: [current metrics]
Expected Output: {
    "alerts": [
        {
            "type": string,
            "severity": string,
            "message": string,
            "timestamp": datetime
        },
        ...
    ]
}
```

## Usage Instructions

1. Select metrics collection scope
2. Configure threshold values
3. Set up monitoring intervals
4. Define alert conditions
5. Establish reporting format

## Best Practices

- Regular metric collection intervals
- Persistent metrics storage
- Alert threshold validation
- Performance impact monitoring
- Historical trend analysis 