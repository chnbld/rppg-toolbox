"""
WebSocket Server for Real-Time rPPG Inference
Faster than REST API - uses persistent connection and binary frames
"""

import asyncio
import sys
import os

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not found.")
    print("Run: pip install opencv-python-headless")
    sys.exit(1)

import numpy as np
import torch
import base64
import json
import scipy
from scipy.signal import butter, filtfilt, periodogram
from config import get_config
from neural_methods.model.DeepPhys import DeepPhys

# Global variables
model = None
device = None
config = None
frame_buffer = []
chunk_size = 150


def calculate_bpm_from_ppg(ppg_signal, fs=30, low_pass=0.6, high_pass=3.3):
    """
    Calculate heart rate in BPM from PPG signal using FFT.
    
    Args:
        ppg_signal: PPG signal array (1D or 2D)
        fs: Sampling rate (frames per second), default=30
        low_pass: Low frequency cutoff in Hz (default 0.6 = 36 BPM)
        high_pass: High frequency cutoff in Hz (default 3.3 = 198 BPM)
    
    Returns:
        BPM value
    """
    # Ensure 2D array
    ppg_signal = np.expand_dims(ppg_signal, 0) if len(ppg_signal.shape) == 1 else ppg_signal
    
    # Find next power of 2 for FFT
    N = 1 if ppg_signal.shape[1] == 0 else 2 ** (ppg_signal.shape[1] - 1).bit_length()
    
    # Compute periodogram (power spectral density)
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    
    # Flatten pxx_ppg to handle 2D output
    pxx_ppg = pxx_ppg.flatten()
    
    # Filter to heart rate range
    mask = (f_ppg >= low_pass) & (f_ppg <= high_pass)
    mask_freq = f_ppg[mask]
    mask_power = pxx_ppg[mask]
    
    if len(mask_freq) == 0:
        return 0.0
    
    # Find dominant frequency
    peak_idx = np.argmax(mask_power)
    dominant_freq = mask_freq[peak_idx]
    
    # Convert to BPM
    bpm = dominant_freq * 60
    
    return float(bpm)


def preprocess_for_bpm(ppg_signal, fs=30):
    """
    Preprocess PPG signal for BPM calculation.
    Apply bandpass filter and detrend.
    
    Args:
        ppg_signal: Raw PPG signal
        fs: Sampling rate
    
    Returns:
        Processed PPG signal
    """
    # Detrend
    signal_length = len(ppg_signal)
    lambda_value = 100
    from scipy.sparse import spdiags
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    from scipy.sparse import spdiags
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    detrended = np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), ppg_signal)
    
    # Bandpass filter [0.6, 3.3] Hz = [36, 198] BPM
    [b, a] = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
    filtered = filtfilt(b, a, np.double(detrended))
    
    return filtered


def initialize_model(config_path):
    """Initialize the DeepPhys model for inference"""
    global model, device, config
    
    # Load config
    import argparse
    from main import add_args
    
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args(['--config_file', config_path])
    
    config = get_config(args)
    device = torch.device(config.DEVICE)
    
    # Initialize model
    img_size = config.TEST.DATA.PREPROCESS.RESIZE.H
    model = DeepPhys(img_size=img_size).to(device)
    
    # Load pretrained weights
    if not os.path.exists(config.INFERENCE.MODEL_PATH):
        raise ValueError(f"Model file not found: {config.INFERENCE.MODEL_PATH}")
    
    # Load state dict
    state_dict = torch.load(config.INFERENCE.MODEL_PATH, map_location=device)
    
    # Remove 'module.' prefix if present (model was saved with DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ Model loaded successfully (img_size={img_size})")
    print(f"✓ Device: {device}")
    print(f"✓ Model path: {config.INFERENCE.MODEL_PATH}")
    
    return model, config


def compute_diff_normalized(frames):
    """Compute diff-normalized frames for DeepPhys"""
    n, h, w, c = frames.shape
    diffnormalized_len = n - 1
    diff_frames = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    
    for j in range(diffnormalized_len):
        diff_frames[j, :, :, :] = (frames[j + 1, :, :, :] - frames[j, :, :, :]) / (
                frames[j + 1, :, :, :] + frames[j, :, :, :] + 1e-7)
    
    # Normalize by std
    diff_frames = diff_frames / (np.std(diff_frames) + 1e-7)
    
    # Pad last frame
    diff_frames_padded = np.append(diff_frames, diff_frames[-1:, :, :, :], axis=0)
    
    return diff_frames_padded


def predict_from_frames(frames, config):
    """Run inference on frames and return predictions"""
    global model, device
    
    # Ensure frames have 3 channels and preprocess
    processed = []
    for frame in frames:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] != 3:
            if frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w = config.TEST.DATA.PREPROCESS.RESIZE.H, config.TEST.DATA.PREPROCESS.RESIZE.W
        resized = cv2.resize(frame, (w, h))
        normalized = resized.astype(np.float32) / 255.0
        processed.append(normalized)
    
    raw_frames = np.array(processed)  # (num_frames, H, W, 3)
    
    # Compute diff-normalized
    diff_frames = compute_diff_normalized(raw_frames)
    
    # Concatenate to get 6 channels (diff + raw)
    combined = np.concatenate([diff_frames, raw_frames], axis=-1)  # (num_frames, H, W, 6)
    
    # Convert to tensor
    tensor = torch.from_numpy(combined).float()  # (num_frames, H, W, 6)
    tensor = tensor.permute(0, 3, 1, 2)  # (num_frames, 6, H, W)
    tensor = tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(tensor)
    
    # Convert to numpy
    predictions_np = predictions.cpu().numpy().flatten()
    
    # Calculate BPM from the prediction
    # The model outputs PPG signal, convert to BPM
    fs = 30  # Video sampling rate (frames per second)
    
    # Preprocess PPG signal
    ppg_processed = preprocess_for_bpm(predictions_np, fs=fs)
    
    # Calculate BPM
    bpm = calculate_bpm_from_ppg(ppg_processed, fs=fs)
    
    return predictions_np, bpm


async def handle_client(websocket, path):
    """Handle WebSocket client connection"""
    global frame_buffer
    
    print(f"Client connected: {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                # Parse message
                data = json.loads(message)
                command = data.get('command')
                
                if command == 'frame':
                    # Receive frame
                    frame_data = data.get('frame')
                    
                    # Decode image
                    if isinstance(frame_data, str):
                        image_bytes = base64.b64decode(frame_data)
                    else:
                        # Assume binary
                        image_bytes = frame_data
                    
                    # Decode frame
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Add to buffer
                        frame_buffer.append(frame_rgb)
                        
                        # Process if we have enough frames
                        if len(frame_buffer) >= chunk_size:
                            try:
                                chunk = np.array(frame_buffer[:chunk_size])
                                predictions, bpm = predict_from_frames(chunk, config)
                                frame_buffer = frame_buffer[chunk_size:]
                                
                                # Send prediction
                                response = {
                                    'status': 'success',
                                    'bpm': bpm,
                                    'prediction': float(np.mean(predictions)),
                                    'predictions': predictions.flatten().tolist(),
                                    'frames_processed': chunk_size
                                }
                                await websocket.send(json.dumps(response))
                            except Exception as e:
                                print(f"Error in prediction: {e}")
                                await websocket.send(json.dumps({
                                    'status': 'error',
                                    'error': str(e)
                                }))
                        else:
                            # Send buffering status
                            response = {
                                'status': 'buffering',
                                'buffer_size': len(frame_buffer),
                                'message': f'Received {len(frame_buffer)}/{chunk_size} frames'
                            }
                            await websocket.send(json.dumps(response))
                    else:
                        await websocket.send(json.dumps({
                            'status': 'error',
                            'error': 'Failed to decode image'
                        }))
                
                elif command == 'reset':
                    frame_buffer = []
                    await websocket.send(json.dumps({'status': 'buffer_reset'}))
                
                elif command == 'status':
                    response = {
                        'status': 'running',
                        'buffer_size': len(frame_buffer),
                        'chunk_size': chunk_size
                    }
                    await websocket.send(json.dumps(response))
                
                else:
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'error': f'Unknown command: {command}'
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'error': 'Invalid JSON'
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'error': str(e)
                }))
                print(f"Error: {e}")
    
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")
    finally:
        print(f"Connection closed: {websocket.remote_address}")


async def main():
    """Start WebSocket server"""
    global model, config
    
    # Get config file
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml'
    
    print("\n" + "=" * 60)
    print("Initializing rPPG WebSocket Server")
    print("=" * 60)
    
    # Initialize model
    model, config = initialize_model(config_path)
    
    print("\n" + "=" * 60)
    print("Starting WebSocket server on ws://localhost:8765")
    print("=" * 60)
    
    # Start server
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())

