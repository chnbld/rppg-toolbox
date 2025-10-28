# WebSocket Server for Real-Time rPPG Inference

## Overview

The WebSocket server provides **faster streaming** compared to the REST API by using persistent connections and efficient binary data transfer. It returns comprehensive cardiovascular metrics including heart rate, respiratory rate, HRV, stress, and workload indices.

## Installation

1. Install dependencies:
```bash
pip install websockets opencv-python-headless
```

2. Run the server:
```bash
python websocket_server.py configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml
```

The server will start on `ws://localhost:8765`

3. In another terminal, test with the example client:
```bash
python websocket_client_example.py
```

## Advantages Over REST API

- **Persistent connection**: No HTTP overhead per frame
- **Binary frame compression**: Frames are compressed before transfer
- **Faster throughput**: Optimal for real-time streaming
- **Lower latency**: Direct WebSocket communication
- **Efficient data transfer**: Gzip compression for responses

## WebSocket Commands

### `{'command': 'status'}`
Check server status

**Response:**
```json
{
    "status": "running",
    "buffer_size": 0,
    "chunk_size": 150
}
```

### `{'command': 'frame', 'frame': base64_encoded_compressed_frame, 'compressed': true}`
Send a video frame

**Response (compressed):**
```json
{
    "status": "success",
    "bpm": 72.5,
    "rr": 15.3,
    "n": 150,
    "sdnn": 45.2,
    "rmssd": 38.5,
    "pnn50": 12.3,
    "st": 35.2,
    "sl": "n",
    "ab": "b",
    "wi": 42.3,
    "wl": "m",
    "rpp": 8320,
    "met": 1.8,
    "mv": 0.1234,
    "sv": 0.0056
}
```

### `{'command': 'reset'}`
Reset the frame buffer

**Response:**
```json
{
    "status": "buffer_reset"
}
```

## Response Fields Explained

### Basic Metrics
- **bpm**: Heart rate in beats per minute
- **rr**: Respiratory rate in breaths per minute
- **n**: Number of frames processed

### HRV (Heart Rate Variability)
- **sdnn**: Standard deviation of RR intervals (ms)
- **rmssd**: Root mean square of successive differences (ms)  
- **pnn50**: Percentage of intervals differing by >50ms (%)

### Cardiac Stress
- **st**: Stress index (0-100)
- **sl**: Stress level ('l'=low, 'n'=normal, 'e'=elevated, 'h'=high)
- **ab**: Autonomic balance ('p'=parasympathetic, 'b'=balanced, 's'=sympathetic)

### Cardiac Workload
- **wi**: Workload index (0-100)
- **wl**: Workload level ('l'=low, 'm'=moderate, 'e'=elevated, 'h'=high)
- **rpp**: Rate-Pressure Product (estimated)
- **met**: Metabolic demand (MET)

### BVP Signal
- **mv**: Mean BVP amplitude
- **sv**: Standard deviation of BVP amplitude

## Usage Example

### Python Client

```python
import asyncio
import websockets
import cv2
import base64
import json
import gzip

async def send_frame(websocket, frame):
    # Resize for efficiency
    small_frame = cv2.resize(frame, (320, 240))
    
    # Encode as JPEG with compression
    _, buffer = cv2.imencode('.jpg', small_frame, 
                              [cv2.IMWRITE_JPEG_QUALITY, 70])
    image_bytes = buffer.tobytes()
    
    # Compress with gzip
    compressed = gzip.compress(image_bytes)
    frame_b64 = base64.b64encode(compressed).decode('utf-8')
    
    # Send to server
    await websocket.send(json.dumps({
        'command': 'frame',
        'frame': frame_b64,
        'compressed': True
    }))
    
    # Receive and decompress response
    response = await websocket.recv()
    if isinstance(response, bytes):
        response = gzip.decompress(response).decode('utf-8')
    
    return json.loads(response)

# Main loop
async def main():
    async with websockets.connect('ws://localhost:8765') as websocket:
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            result = await send_frame(websocket, frame)
            
            if result['status'] == 'success':
                print(f"HR: {result['bpm']:.1f} BPM")
                print(f"RR: {result['rr']:.1f} breaths/min")
                print(f"Stress: {result['st']:.1f}/100 ({result['sl']})")
                print(f"Workload: {result['wi']:.1f}/100 ({result['wl']})")
                print(f"SDNN: {result['sdnn']:.1f} ms")
                print(f"RMSSD: {result['rmssd']:.1f} ms")
                print(f"RPP: {result['rpp']}")
                print(f"MET: {result['met']:.1f}")
                print()

asyncio.run(main())
```

## How It Works

### 1. Frame Processing
- Frames are resized to 320x240 on client side
- JPEG encoded with quality 70
- Compressed with gzip
- Base64 encoded and sent

### 2. Server Processing
- Frames decompressed and decoded
- Accumulated in buffer (150 frames)
- Processed through DeepPhys model
- Returns comprehensive metrics

### 3. Response Format
- JSON compressed with gzip
- Minimal field names (e.g., 'bpm', 'rr', 'st')
- Only essential metrics sent
- ~200-300 bytes per response

## Performance

- **Frame Rate**: Up to 30 FPS
- **Latency**: ~50-100ms per chunk processing
- **Bandwidth**: ~2-5 KB per frame (compressed)
- **Memory**: Efficient streaming with chunked processing
- **CPU Usage**: Optimized for CPU-only inference

## Metrics Interpretation

### Heart Rate (BPM)
- Normal resting: 60-100 BPM
- Elevated: >100 BPM (exercise, stress)
- Low: <60 BPM (athletes, rest)

### Respiratory Rate
- Normal: 12-20 breaths/min
- Elevated: >20 breaths/min (physical activity)

### HRV Metrics

#### SDNN (Standard Deviation of RR Intervals)
- **High (30-50ms)**: Good autonomic function, low stress
- **Low (<20ms)**: Reduced variability, possible stress/illness

#### RMSSD (Heart Rate Variability)
- **High (>30ms)**: Strong parasympathetic (rest/digest) activity
- **Low (<20ms)**: Dominant sympathetic (fight/flight) activity

#### pNN50
- **High (10-30%)**: Good adaptability, healthy autonomic function
- **Low (<5%)**: Reduced variability, possible health issues

### Cardiac Stress Index
- **0-25**: Low stress (relaxed, calm)
- **25-50**: Normal stress levels
- **50-75**: Elevated stress
- **75-100**: High stress (may need attention)

### Cardiac Workload
- **0-25**: Low workload (resting)
- **25-50**: Moderate (light activity)
- **50-75**: Elevated (moderate activity)
- **75-100**: High workload (intense activity)

### Autonomic Balance
- **Parasympathetic Dominant**: Calm, relaxed state
- **Balanced**: Normal state
- **Sympathetic Dominant**: Stressed, fight-or-flight response

### Rate-Pressure Product (RPP)
- Estimate of cardiac workload
- RPP = Heart Rate × Systolic Blood Pressure
- Higher RPP = More cardiac work
- Normal at rest: <10,000
- Max exercise: 40,000+

### Metabolic Demand (MET)
- 1 MET = Resting metabolic rate
- 1.5 MET = Light activity
- 2.5 MET = Moderate activity
- 3.5+ MET = Vigorous activity

## Configuration

Configuration is loaded from your YAML file:
- Image size: Set in `RESIZE.H` and `RESIZE.W` in config
- Chunk size: 150 frames (fixed for optimal processing)
- Device: CPU or CUDA (set in config)

## Troubleshooting

### "Message too big" Error
The frames are automatically compressed. If you still get this error:
- Reduce video resolution further in client
- Lower JPEG quality in `websocket_client_example.py`
- Increase chunk size in server configuration

### Connection Issues
```bash
# Check if server is running
curl http://localhost:8765

# Or test with websocat
websocat ws://localhost:8765
```

### Performance Optimization
- Use lower resolution input (320x240 recommended)
- Reduce chunk size in config if memory is limited
- Set `num_workers=0` for CPU-only inference

## Comparison with Flask REST API

| Feature | WebSocket | REST API |
|---------|-----------|----------|
| Connection | Persistent | Request-based |
| Latency | Lower | Higher |
| Throughput | Higher | Lower |
| Compression | Gzip | None |
| Frame Handling | Binary optimized | JSON |
| Best For | Real-time streaming | Simple HTTP integration |
| Complexity | Medium | Low |

## Notes

- All metrics calculated from BVP (Blood Volume Pulse) signal
- Research-grade estimates, not medical devices
- Designed for real-time monitoring and analysis
- Supports long-running connections (hours)
- Automatic reconnection recommended for production use
- Frame buffer accumulates 150 frames before processing
- Each prediction includes all metrics (HR, RR, HRV, stress, workload)

