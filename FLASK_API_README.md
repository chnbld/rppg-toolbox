# Flask API Server for Real-Time rPPG Inference

## Overview

This Flask API server receives video frames via HTTP requests and returns pulse predictions using the DeepPhys model. It streams frames into a buffer and processes them in chunks.

## Quick Fix Required

**IMPORTANT:** There's a typo in `flask_api_server.py`. Please fix the NumPy function name:

Find all instances of `np.frombuffer` and replace with `np.frombuffer`

This occurs on lines:
- 217
- 279  
- 337

## Installation

1. Install Flask if not already installed:
```bash
pip install flask requests
```

2. Run the server:
```bash
python flask_api_server.py configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml
```

3. In another terminal, test with the example client:
```bash
python flask_client_example.py
```

## API Endpoints

### GET /
Home page with API documentation

### GET /status
Check server status

**Response:**
```json
{
    "status": "running",
    "model_loaded": true,
    "buffer_size": 150,
    "device": "cpu",
    "chunk_size": 150
}
```

### POST /reset
Reset the frame buffer

### POST /infer_frame
Send a single frame via multipart form data

**Request:**
- Content-Type: `multipart/form-data`
- Body: `{"frame": <image file>}`

**Response:**
```json
{
    "status": "success",
    "prediction": 0.0123,
    "predictions": [0.0121, 0.0122, ...],
    "frames_processed": 150,
    "buffer_remaining": 0
}
```

### POST /infer_frame_base64
Send a frame as base64 encoded string

**Request:**
```json
{
    "frame": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
    "status": "success",
    "prediction": 0.0123,
    "predictions": [...],
    "frames_processed": 150,
    "buffer_remaining": 0
}
```

## Usage Examples

### Python Example

```python
import requests
import cv2
import base64

# Capture frames from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Encode frame as base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Send to API
    response = requests.post(
        'http://localhost:5000/infer_frame_base64',
        json={'frame': frame_b64}
    )
    
    result = response.json()
    if result['status'] == 'success':
        print(f"Pulse prediction: {result['prediction']:.4f}")
```

### cURL Example

```bash
# Send a frame
curl -X POST http://localhost:5000/infer_frame \
  -F "frame=@path/to/image.jpg"

# Check status
curl http://localhost:5000/status

# Reset buffer
curl -X POST http://localhost:5000/reset
```

## How It Works

1. **Frame Buffer**: Frames are accumulated in a buffer until you have enough (default: 150 frames)
2. **Processing**: Once buffer is full, frames are:
   - Resized to configured size (e.g., 72x72)
   - Compute diff-normalized version
   - Concatenate diff + raw to get 6 channels
   - Run through DeepPhys model
3. **Return**: Results are returned immediately

## Configuration

The server uses your existing config file:
- Image size: Set in `RESIZE.H` and `RESIZE.W` in config YAML
- Chunk size: 150 frames (hardcoded in server)

## Notes

- The buffer accumulates frames until reaching chunk_size (150)
- Each prediction processes 150 frames
- Memory usage is optimized for streaming
- CPU-only inference supported

