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


def calculate_cardiac_stress_index(hrv, heart_rate):
    """
    Calculate cardiac stress index from HRV and heart rate metrics.
    
    Cardiac stress is typically indicated by:
    - Low HRV (low RMSSD, low SDNN)
    - Elevated heart rate relative to resting
    - Low pNN50 (reduced variability)
    
    Args:
        hrv: Dictionary with HRV metrics
        heart_rate: Current heart rate in BPM
    
    Returns:
        Dictionary with stress metrics:
        - stress_index: 0-100 scale (0 = low stress, 100 = high stress)
        - stress_level: 'low', 'normal', 'elevated', 'high'
        - autonomic_balance: Parasympathetic activity indicator
    """
    sdnn = hrv.get('sdnn', 0)
    rmssd = hrv.get('rmssd', 0)
    pnn50 = hrv.get('pnn50', 0)
    mean_hr = hrv.get('mean_hr', heart_rate)
    
    # Calculate stress indicators (normalized to 0-100)
    # Lower HRV = Higher stress
    
    # SDNN component (inverted: lower SDNN = higher stress)
    if sdnn > 50:
        sdnn_score = 0  # Very low stress
    elif sdnn > 35:
        sdnn_score = 20  # Low stress
    elif sdnn > 20:
        sdnn_score = 50  # Moderate stress
    else:
        sdnn_score = 80  # High stress
    
    # RMSSD component (inverted: lower RMSSD = higher stress)
    if rmssd > 40:
        rmssd_score = 0  # Very low stress
    elif rmssd > 25:
        rmssd_score = 25  # Low stress
    elif rmssd > 15:
        rmssd_score = 55  # Moderate stress
    else:
        rmssd_score = 85  # High stress
    
    # Heart rate component (elevated HR indicates stress)
    hr_deviation = max(0, mean_hr - 65)  # Deviation from baseline
    if hr_deviation > 20:
        hr_score = 80  # Significantly elevated
    elif hr_deviation > 10:
        hr_score = 40  # Moderately elevated
    else:
        hr_score = 10  # Normal
    
    # Weighted average to get overall stress index
    stress_index = (sdnn_score * 0.4 + rmssd_score * 0.4 + hr_score * 0.2)
    
    # Determine stress level
    if stress_index < 25:
        stress_level = 'low'
    elif stress_index < 50:
        stress_level = 'normal'
    elif stress_index < 75:
        stress_level = 'elevated'
    else:
        stress_level = 'high'
    
    # Autonomic balance indicator
    if rmssd > 30:
        balance = 'parasympathetic_dominant'  # Calm, relaxed
    elif rmssd > 15:
        balance = 'balanced'
    else:
        balance = 'sympathetic_dominant'  # Stressed, fight-or-flight
    
    return {
        'stress_index': float(stress_index),
        'stress_level': stress_level,
        'autonomic_balance': balance,
        'components': {
            'sdnn_component': float(sdnn_score),
            'rmssd_component': float(rmssd_score),
            'heart_rate_component': float(hr_score)
        }
    }


def calculate_cardiac_workload(heart_rate, stress_index, respiratory_rate):
    """
    Calculate cardiac workload (myocardial work) from physiological metrics.
    
    Cardiac workload represents the amount of work the heart performs per minute.
    Higher workload indicates increased myocardial oxygen consumption.
    
    Args:
        heart_rate: Current heart rate in BPM
        stress_index: Stress index (0-100)
        respiratory_rate: Respiratory rate in breaths/min
    
    Returns:
        Dictionary with workload metrics:
        - workload_index: Normalized workload (0-100)
        - workload_level: 'low', 'moderate', 'elevated', 'high'
        - estimated_rpp: Rate-Pressure Product approximation
        - metabolic_demand: Estimated metabolic demand
    """
    # Base workload from heart rate
    # Resting HR ~60-70 bpm, Max typically 220 - age
    # Workload scales with HR elevation above resting
    resting_hr = 65
    hr_excess = max(0, heart_rate - resting_hr)
    
    # Normalize to 0-100 scale
    # Assuming max HR of ~150 for typical adult (moderate exercise)
    hr_workload = min(100, (hr_excess / 85) * 100)
    
    # Add stress component (stress increases workload)
    stress_factor = stress_index / 100.0
    stress_workload = stress_factor * 30
    
    # Add respiratory component (higher RR may indicate increased metabolic demand)
    rr_factor = max(0, min(1, (respiratory_rate - 12) / 20))  # Normal ~12-20
    respiratory_workload = rr_factor * 20
    
    # Total workload (weighted combination)
    workload_index = (hr_workload * 0.6 + stress_workload * 0.25 + respiratory_workload * 0.15)
    
    # Determine workload level
    if workload_index < 25:
        workload_level = 'low'
    elif workload_index < 50:
        workload_level = 'moderate'
    elif workload_index < 75:
        workload_level = 'elevated'
    else:
        workload_level = 'high'
    
    # Estimate Rate-Pressure Product (RPP)
    # RPP = HR × SBP (typical measure of cardiac work)
    # Without SBP, estimate based on HR and stress
    # Typical SBP: 100-140 mmHg, rough estimate based on stress
    estimated_sbp = 115 + (stress_index * 0.25)  # Baseline + stress contribution
    estimated_rpp = heart_rate * estimated_sbp
    
    # Metabolic equivalent (MET) approximation
    # 1 MET = resting metabolic rate
    # Estimated from HR: 0.5-1.0 MET at rest, up to 3-4 MET with elevated HR
    if heart_rate < 70:
        met_estimate = 0.8
    elif heart_rate < 90:
        met_estimate = 1.2
    elif heart_rate < 110:
        met_estimate = 1.8
    elif heart_rate < 130:
        met_estimate = 2.5
    else:
        met_estimate = 3.5
    
    return {
        'workload_index': float(workload_index),
        'workload_level': workload_level,
        'estimated_rpp': float(estimated_rpp),
        'metabolic_demand': float(met_estimate),
        'components': {
            'heart_rate_component': float(hr_workload),
            'stress_component': float(stress_workload),
            'respiratory_component': float(respiratory_workload)
        }
    }


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
    
    # Calculate cardiac stress index
    stress = calculate_cardiac_stress_index(hrv, bpm)
    
    # Calculate cardiac workload
    workload = calculate_cardiac_workload(bpm, stress['stress_index'], rr)
    
    return predictions_np, bpm, rr, hrv, stress, workload


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
                                predictions, bpm, rr, hrv, stress, workload = predict_from_frames(chunk, config)
                                frame_buffer = frame_buffer[chunk_size:]
                                
                                # Send prediction with HRV, stress, and workload metrics
                                # Note: 'predictions' is the BVP (Blood Volume Pulse) signal
                                # Limit BVP signal to prevent WebSocket message size issues
                                # Send only summary stats, not full signal array
                                bvp_signal_summary = predictions.flatten().tolist()[:50]  # First 50 points only
                                
                                response = {
                                    'status': 'success',
                                    'bpm': bpm,
                                    'respiratory_rate': rr,
                                    'hrv': hrv,
                                    'cardiac_stress': stress,
                                    'cardiac_workload': workload,
                                    'bvp_signal_sample': bvp_signal_summary,  # Sample only
                                    'bvp_mean_amplitude': float(np.mean(predictions)),
                                    'bvp_std_amplitude': float(np.std(predictions)),
                                    'frames_processed': chunk_size,
                                    # For backward compatibility
                                    'prediction': float(np.mean(predictions)),
                                    'prediction_sample': bvp_signal_summary
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

