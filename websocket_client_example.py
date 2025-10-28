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

API_URL = "ws://localhost:8765"


async def send_frame_async(websocket, frame):
    """Send a frame via WebSocket"""
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    
    # Encode as base64
    frame_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Send as JSON
    message = {
        'command': 'frame',
        'frame': frame_b64
    }
    
    await websocket.send(json.dumps(message))
    
    # Wait for response
    response = await websocket.recv()
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
            video_path = "/home/ubuntu/test.mp4"  # Update this
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
                        bpm = result.get('bpm', 0)
                        rr = result.get('respiratory_rate', 0)
                        hrv = result.get('hrv', {})
                        stress = result.get('cardiac_stress', {})
                        workload = result.get('cardiac_workload', {})
                        print(f"\n✓ BVP Signal Analysis:")
                        print(f"  Mean Amplitude: {result.get('bvp_mean_amplitude', 0):.4f}")
                        print(f"  Std Amplitude: {result.get('bvp_std_amplitude', 0):.4f}")
                        print(f"  Signal Length: {result.get('frames_processed', 0)} points (sample shown)")
                        print(f"  Heart Rate: {bpm:.2f} BPM")
                        print(f"  Respiratory Rate: {rr:.2f} breaths/min")
                        if hrv:
                            print(f"  HRV:")
                            print(f"    SDNN: {hrv.get('sdnn', 0):.2f} ms")
                            print(f"    RMSSD: {hrv.get('rmssd', 0):.2f} ms")
                            print(f"    pNN50: {hrv.get('pnn50', 0):.2f}%")
                        if stress:
                            print(f"  Cardiac Stress: {stress.get('stress_index', 0):.1f}/100 ({stress.get('stress_level', 'unknown')})")
                            print(f"    Autonomic Balance: {stress.get('autonomic_balance', 'unknown')}")
                        if workload:
                            print(f"  Cardiac Workload: {workload.get('workload_index', 0):.1f}/100 ({workload.get('workload_level', 'unknown')})")
                            print(f"    Estimated RPP: {workload.get('estimated_rpp', 0):.0f}")
                            print(f"    Metabolic Demand: {workload.get('metabolic_demand', 0):.1f} MET")
                        print(f"  Frames processed: {result['frames_processed']}")
                        predictions.append(result.get('bvp_mean_amplitude', result['prediction']))
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

