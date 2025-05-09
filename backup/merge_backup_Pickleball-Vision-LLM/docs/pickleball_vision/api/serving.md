# API Serving

The Pickleball Vision API provides HTTP endpoints for ball detection and tracking.

## Overview

The API is built with FastAPI and provides:
- Ball detection endpoint
- Metrics endpoint
- Health check endpoint

## Endpoints

### Ball Detection

```http
POST /detect
Content-Type: multipart/form-data

file: video_file
```

Processes a video file and returns detections:

```json
{
  "detections": [
    {
      "frame_id": 0,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ],
  "metrics": {
    "confidence": 0.95,
    "frames_processed": 100,
    "detections_found": 50
  }
}
```

### Metrics

```http
GET /metrics
```

Returns current metrics:

```json
{
  "confidence": [0.95, 0.92, 0.97],
  "quality": [0.85, 0.82, 0.88]
}
```

### Health Check

```http
GET /health
```

Returns API health status:

```json
{
  "status": "healthy"
}
```

## Components

The API integrates several components:

### Configuration
- Loads settings from YAML
- Configures model parameters
- Sets output directories

### Preprocessing
- Resizes frames
- Checks frame quality
- Computes motion scores

### Detection
- Runs ball detection model
- Returns bounding boxes
- Provides confidence scores

### Monitoring
- Records metrics
- Tracks performance
- Enables analysis

## Usage

Start the API server:

```bash
uvicorn pickleball_vision.api.serving.app:app --host 0.0.0.0 --port 8000
```

Send a request:

```python
import requests

files = {"video": open("game.mp4", "rb")}
response = requests.post("http://localhost:8000/detect", files=files)
detections = response.json()
``` 