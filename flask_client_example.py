"""
Example client for Flask rPPG API
Demonstrates how to send frames and receive predictions
"""

import requests
import cv2
import base64
import numpy as np
import time

API_BASE_URL = "http://localhost:5000"

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    
    # Convert to base64
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    return base64_str


def send_frame_multipart(frame_path):
    """Send frame using multipart/form-data"""
    url = f"{API_BASE_URL}/infer_frame"
    
    with open(frame_path, 'rb') as frame_file:
        files = {'frame': frame_file}
        response = requests.post(url, files=files)
    
    return response.json()


def send_frame_base64(frame):
    """Send frame as base64 JSON"""
    url = f"{API_BASE_URL}/infer_frame_base64"
    
    # Convert frame to base64
    frame_b64 = frame_to_base64(frame)
    
    # Send JSON request
    response = requests.post(url, json={'frame': frame_b64})
    
    return response.json()


def check_status():
    """Check server status"""
    response = requests.get(f"{API_BASE_URL}/status")
    return response.json()


def reset_buffer():
    """Reset frame buffer"""
    response = requests.post(f"{API_BASE_URL}/reset")
    return response.json()


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("rPPG API Client Example")
    print("=" * 60)
    
    # Check server status
    print("\n1. Checking server status...")
    status = check_status()
    print(f"   Status: {status}")
    
    # Example: Send frames from a video
    print("\n2. Sending frames from video...")
    video_path = "RAWDATA/s1/vid_s1_T1.avi"  # Update this to your video path
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_predictions = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Send frame
            result = send_frame_base64(frame)
            
            if result['status'] == 'success':
                print(f"\n✓ Got prediction! Mean: {result['prediction']:.4f}")
                total_predictions.append(result['prediction'])
                print(f"   Frames processed: {result['frames_processed']}")
                print(f"   Buffer remaining: {result['buffer_remaining']}")
            elif result['status'] == 'buffering':
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"   Buffering: {result['buffer_size']}/150 frames...")
            
            frame_count += 1
            time.sleep(0.033)  # Simulate ~30 FPS
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        cap.release()
    
    if total_predictions:
        print(f"\n✓ Received {len(total_predictions)} predictions")
        print(f"   Mean prediction: {np.mean(total_predictions):.4f}")
        print(f"   Std prediction: {np.std(total_predictions):.4f}")
    
    print("\n" + "=" * 60)

