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
                        response_data = json.loads(decompressed.decode('utf-8'))
                    except:
                        response_data = json.loads(response)
                    
                    # Print results if we got predictions
                    if response_data.get('status') == 'success':
                        print(f"\n✓ Prediction Results (Frame {frame_count}):")
                        print(f"  Heart Rate: {response_data.get('bpm', 0):.1f} BPM")
                        print(f"  Respiratory Rate: {response_data.get('rr', 0):.1f} breaths/min")
                        print(f"  SDNN: {response_data.get('sdnn', 0):.2f} ms")
                        print(f"  RMSSD: {response_data.get('rmssd', 0):.2f} ms")
                        print(f"  pNN50: {response_data.get('pnn50', 0):.2f}%")
                        print(f"  Stress Index: {response_data.get('st', 0):.1f}")
                        print(f"  Stress Level: {response_data.get('sl', 'normal')}")
                        print(f"  Workload Index: {response_data.get('wi', 0):.1f}")
                        print(f"  Workload Level: {response_data.get('wl', 'moderate')}")
                        print(f"  Estimated RPP: {response_data.get('rpp', 0)}")
                        print(f"  Metabolic Demand: {response_data.get('met', 0):.1f} MET")
                    elif response_data.get('status') == 'buffering':
                        print(f"\r  Buffering: {response_data.get('buffer_size', 0)}/{response_data.get('chunk_size', 128)} frames", end='')
                    elif response_data.get('status') == 'error':
                        print(f"\n  Error: {response_data.get('error', 'Unknown error')}")
                
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

