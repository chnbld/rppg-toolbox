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


def calculate_respiratory_rate_from_ppg(ppg_signal, fs=30, low_pass=0.13, high_pass=0.5):
    """
    Calculate respiratory rate in breaths per minute from PPG signal using FFT.
    
    Args:
        ppg_signal: PPG signal array (1D or 2D)
        fs: Sampling rate (frames per second), default=30
        low_pass: Low frequency cutoff in Hz (default 0.13 = 8 breaths/min)
        high_pass: High frequency cutoff in Hz (default 0.5 = 30 breaths/min)
    
    Returns:
        Respiratory rate value in breaths per minute
    """
    # Ensure 2D array
    ppg_signal = np.expand_dims(ppg_signal, 0) if len(ppg_signal.shape) == 1 else ppg_signal
    
    # Find next power of 2 for FFT
    N = 1 if ppg_signal.shape[1] == 0 else 2 ** (ppg_signal.shape[1] - 1).bit_length()
    
    # Compute periodogram (power spectral density)
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    
    # Flatten pxx_ppg to handle 2D output
    pxx_ppg = pxx_ppg.flatten()
    
    # Filter to respiratory rate range
    mask = (f_ppg >= low_pass) & (f_ppg <= high_pass)
    mask_freq = f_ppg[mask]
    mask_power = pxx_ppg[mask]
    
    if len(mask_freq) == 0:
        return 0.0
    
    # Find dominant frequency
    peak_idx = np.argmax(mask_power)
    dominant_freq = mask_freq[peak_idx]
    
    # Convert to breaths per minute
    rr = dominant_freq * 60
    
    return float(rr)


def calculate_hrv_from_ppg(ppg_signal, fs=30):
    """
    Calculate Heart Rate Variability (HRV) metrics from PPG signal.
    
    Args:
        ppg_signal: PPG signal array
        fs: Sampling rate (frames per second), default=30
    
    Returns:
        Dictionary with HRV metrics:
        - sdnn: Standard deviation of RR intervals (ms)
        - rmssd: Root mean square of successive differences (ms)
        - pnn50: Percentage of intervals differing by >50ms (%)
        - mean_rr: Mean RR interval (ms)
        - mean_hr: Mean heart rate (bpm)
    """
    # Find peaks in the PPG signal
    from scipy.signal import find_peaks
    
    # Detect peaks with minimum distance
    min_distance = int(fs * 0.6)  # At least 0.6s between peaks (max 100 bpm)
    peaks, properties = find_peaks(ppg_signal, distance=min_distance)
    
    if len(peaks) < 2:
        # Not enough peaks for HRV calculation
        return {
            'sdnn': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0,
            'mean_rr': 0.0,
            'mean_hr': 0.0
        }
    
    # Calculate RR intervals (time between peaks in milliseconds)
    rr_intervals = np.diff(peaks) / fs * 1000  # Convert to ms
    
    # Remove outliers (RR intervals that are too short or too long)
    # Filter out intervals < 400ms (HR > 150) or > 2000ms (HR < 30)
    valid_intervals = rr_intervals[(rr_intervals >= 400) & (rr_intervals <= 2000)]
    
    if len(valid_intervals) < 2:
        return {
            'sdnn': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0,
            'mean_rr': np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.0,
            'mean_hr': 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.0
        }
    
    # Calculate time-domain HRV metrics
    mean_rr = np.mean(valid_intervals)
    sdnn = np.std(valid_intervals)
    
    # Calculate RMSSD (Root Mean Square of Successive Differences)
    if len(valid_intervals) > 1:
        differences = np.diff(valid_intervals)
        rmssd = np.sqrt(np.mean(differences ** 2))
    else:
        rmssd = 0.0
    
    # Calculate pNN50 (percentage of adjacent RR intervals differing by >50ms)
    if len(valid_intervals) > 1:
        nn50_count = np.sum(np.abs(differences) > 50)
        pnn50 = (nn50_count / len(differences)) * 100 if len(differences) > 0 else 0.0
    else:
        pnn50 = 0.0
    
    # Calculate mean heart rate
    mean_hr = 60000 / mean_rr if mean_rr > 0 else 0.0
    
    return {
        'sdnn': float(sdnn),
        'rmssd': float(rmssd),
        'pnn50': float(pnn50),
        'mean_rr': float(mean_rr),
        'mean_hr': float(mean_hr)
    }


def preprocess_for_rr(ppg_signal, fs=30):
    """
    Preprocess PPG signal for respiratory rate calculation.
    Apply bandpass filter and detrend.
    
    Args:
        ppg_signal: Raw PPG signal
        fs: Sampling rate
    
    Returns:
        Processed PPG signal for respiratory rate
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
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    detrended = np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), ppg_signal)
    
    # Bandpass filter [0.13, 0.5] Hz = [8, 30] breaths per minute
    [b, a] = butter(1, [0.13 / fs * 2, 0.5 / fs * 2], btype='bandpass')
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
    
    # Calculate BPM and Respiratory Rate from the prediction
    # The model outputs PPG signal, convert to BPM and RR
    fs = 30  # Video sampling rate (frames per second)
    
    # Preprocess PPG signal for heart rate
    ppg_processed_hr = preprocess_for_bpm(predictions_np, fs=fs)
    bpm = calculate_bpm_from_ppg(ppg_processed_hr, fs=fs)
    
    # Preprocess PPG signal for respiratory rate
    ppg_processed_rr = preprocess_for_rr(predictions_np, fs=fs)
    rr = calculate_respiratory_rate_from_ppg(ppg_processed_rr, fs=fs)
    
    # Calculate HRV metrics from the preprocessed signal
    hrv = calculate_hrv_from_ppg(ppg_processed_hr, fs=fs)
    
    return predictions_np, bpm, rr, hrv


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
                                predictions, bpm, rr, hrv = predict_from_frames(chunk, config)
                                frame_buffer = frame_buffer[chunk_size:]
                                
                                # Send prediction with HRV metrics
                                # Note: 'predictions' is the BVP (Blood Volume Pulse) signal
                                response = {
                                    'status': 'success',
                                    'bpm': bpm,
                                    'respiratory_rate': rr,
                                    'hrv': hrv,
                                    'bvp_signal': predictions.flatten().tolist(),
                                    'bvp_mean_amplitude': float(np.mean(predictions)),
                                    'bvp_std_amplitude': float(np.std(predictions)),
                                    'frames_processed': chunk_size,
                                    # For backward compatibility
                                    'prediction': float(np.mean(predictions)),
                                    'predictions': predictions.flatten().tolist()
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

