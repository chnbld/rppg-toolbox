"""
Example WebSocket client for rPPG inference
Demonstrates faster frame streaming compared to REST API
"""

import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
import time
import gzip

API_URL = "ws://localhost:8765"


async def send_frame_async(websocket, frame):
    """Send a frame via WebSocket"""
    # Resize frame to reduce size (max 320x240)
    small_frame = cv2.resize(frame, (320, 240))
    
    # Encode frame as JPEG with lower quality (70 for smaller size)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
    _, buffer = cv2.imencode('.jpg', small_frame, encode_params)
    image_bytes = buffer.tobytes()
    
    # Compress using gzip before encoding
    compressed_bytes = gzip.compress(image_bytes)
    
    # Encode as base64
    frame_b64 = base64.b64encode(compressed_bytes).decode('utf-8')
    
    # Send as JSON
    message = {
        'command': 'frame',
        'frame': frame_b64,
        'compressed': True  # Flag to indicate compression
    }
    
    await websocket.send(json.dumps(message))
    
    # Wait for response and decompress if needed
    response = await websocket.recv()
    
    # Try to decompress (compressed responses are bytes)
    if isinstance(response, bytes):
        try:
            response = gzip.decompress(response).decode('utf-8')
        except:
            # Not compressed, treat as string
            response = response.decode('utf-8') if isinstance(response, bytes) else response
    
    return json.loads(response)


async def reset_buffer(websocket):
    """Reset frame buffer"""
    message = {'command': 'reset'}
    await websocket.send(json.dumps(message))
    response = await websocket.recv()
    return json.loads(response)


async def check_status(websocket):
    """Check server status"""
    message = {'command': 'status'}
    await websocket.send(json.dumps(message))
    response = await websocket.recv()
    return json.loads(response)


async def main():
    """Main client loop"""
    print("=" * 60)
    print("rPPG WebSocket Client")
    print("=" * 60)
    
    try:
        # Connect to server
        print(f"\nConnecting to {API_URL}...")
        async with websockets.connect(API_URL) as websocket:
            print("✓ Connected!")
            
            # Check status
            status = await check_status(websocket)
            print(f"Server status: {status}")
            
            # Example: Send frames from a video
            video_path = "/home/ubuntu/RAWDATA/s1/vid_s1_T1.avi"  # Update this
            print(f"\nProcessing video: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            predictions = []
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Send frame
                    result = await send_frame_async(websocket, frame)
                    
                    if result.get('status') == 'success':
                        # Map short keys to display values
                        bpm = result.get('bpm', 0)
                        rr = result.get('rr', 0)
                        hrv_sdnn = result.get('sdnn', 0)
                        hrv_rmssd = result.get('rmssd', 0)
                        hrv_pnn50 = result.get('pnn50', 0)
                        stress_idx = result.get('st', 0)
                        stress_lvl = result.get('sl', 'n')
                        auton_bal = result.get('ab', 'b')
                        work_idx = result.get('wi', 0)
                        work_lvl = result.get('wl', 'm')
                        rpp = result.get('rpp', 0)
                        met = result.get('met', 0)
                        bvp_mean = result.get('mv', 0)
                        bvp_std = result.get('sv', 0)
                        n_frames = result.get('n', 0)
                        
                        print(f"\n✓ BVP Signal Analysis:")
                        print(f"  BVP Mean: {bvp_mean:.4f}")
                        print(f"  BVP Std: {bvp_std:.4f}")
                        print(f"  Signal Length: {n_frames} points")
                        print(f"  Heart Rate: {bpm:.2f} BPM")
                        print(f"  Respiratory Rate: {rr:.2f} breaths/min")
                        print(f"  HRV:")
                        print(f"    SDNN: {hrv_sdnn:.2f} ms")
                        print(f"    RMSSD: {hrv_rmssd:.2f} ms")
                        print(f"    pNN50: {hrv_pnn50:.2f}%")
                        print(f"  Cardiac Stress: {stress_idx:.1f}/100 ({stress_lvl})")
                        print(f"    Autonomic Balance: {auton_bal}")
                        print(f"  Cardiac Workload: {work_idx:.1f}/100 ({work_lvl})")
                        print(f"    Estimated RPP: {rpp:.0f}")
                        print(f"    Metabolic Demand: {met:.1f} MET")
                        predictions.append(bvp_mean)
                    elif result.get('status') == 'buffering':
                        if frame_count % 30 == 0:
                            print(f"Buffering: {result['buffer_size']}/150 frames...")
                    elif result.get('status') == 'error':
                        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
                        break
                    
                    frame_count += 1
                    await asyncio.sleep(0.033)  # ~30 FPS
                    
            except KeyboardInterrupt:
                print("\n\nStopped by user")
            finally:
                cap.release()
            
            if predictions:
                print(f"\n✓ Received {len(predictions)} predictions")
                print(f"  Mean: {np.mean(predictions):.4f}")
                print(f"  Std: {np.std(predictions):.4f}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

