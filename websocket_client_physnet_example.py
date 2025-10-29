"""
Example WebSocket client for PhysNet rPPG inference
This client sends frames to the PhysNet WebSocket server and receives predictions
"""

import asyncio
import websockets
import cv2
import base64
import json
import gzip


async def send_frame(websocket, frame, compressed=True):
    """Send a single frame to the server"""
    # Resize frame to reduce message size
    frame_resized = cv2.resize(frame, (320, 240))
    
    # Encode as JPEG with lower quality
    _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpeg_data = buffer.tobytes()
    
    # Compress the JPEG data
    if compressed:
        compressed_data = gzip.compress(jpeg_data)
        frame_b64 = base64.b64encode(compressed_data).decode('utf-8')
    else:
        frame_b64 = base64.b64encode(jpeg_data).decode('utf-8')
    
    # Send frame
    message = {
        'command': 'frame',
        'frame': frame_b64,
        'compressed': compressed
    }
    await websocket.send(json.dumps(message))


async def process_video(video_path):
    """Process a video file and send frames to the WebSocket server"""
    print(f"Connecting to ws://localhost:8765...")
    
    async with websockets.connect("ws://localhost:8765") as websocket:
        print("✓ Connected!")
        
        # Get server status
        await websocket.send(json.dumps({'command': 'status'}))
        response = json.loads(await websocket.recv())
        print(f"Server status: {response}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        
        print(f"\nProcessing video: {video_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Send frame
                await send_frame(websocket, frame)
                
                # Try to receive response (non-blocking)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    
                    # Decompress if needed
                    try:
                        decompressed = gzip.decompress(response)
                        result = json.loads(decompressed.decode('utf-8'))
                    except:
                        result = json.loads(response)
                    
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
                
                except asyncio.TimeoutError:
                    # No response yet, continue
                    pass
                
        except KeyboardInterrupt:
            print("\n\nStopping video processing...")
        finally:
            cap.release()
            print(f"\n✓ Processed {frame_count} frames")
            print("✓ Disconnected")


async def test_connection():
    """Test the WebSocket connection"""
    print("Connecting to ws://localhost:8765...")
    
    async with websockets.connect("ws://localhost:8765") as websocket:
        print("✓ Connected!")
        
        # Get server status
        await websocket.send(json.dumps({'command': 'status'}))
        response = json.loads(await websocket.recv())
        print(f"Server status: {response}")
        
        # Test reset command
        await websocket.send(json.dumps({'command': 'reset'}))
        response = json.loads(await websocket.recv())
        print(f"Reset response: {response}")


async def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        # Process video file
        video_path = sys.argv[1]
        await process_video(video_path)
    else:
        # Test connection
        await test_connection()


if __name__ == "__main__":
    asyncio.run(main())

