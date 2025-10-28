"""
Flask API Server for Real-Time rPPG Inference
Receives video frames and returns pulse predictions
"""

from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import torch
import base64
import io
from PIL import Image
import os
import sys
from config import get_config
from neural_methods.model.DeepPhys import DeepPhys

app = Flask(__name__)

# Global variables
model = None
device = None
config = None
frame_buffer = []
chunk_size = 150

# HTML template for testing
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>rPPG Inference API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
        .endpoint { background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }
        code { background: #eee; padding: 2px 5px; }
    </style>
</head>
<body>
    <h1>rPPG Inference API Server</h1>
    <p>Server is running on <code>http://localhost:5000</code></p>
    
    <h2>Available Endpoints:</h2>
    
    <div class="endpoint">
        <h3>POST /infer_frame</h3>
        <p>Send a single frame and get prediction</p>
        <p><strong>Content-Type:</strong> multipart/form-data</p>
        <p><strong>Body:</strong> {"frame": [image file]}</p>
    </div>
    
    <div class="endpoint">
        <h3>POST /infer_frame_base64</h3>
        <p>Send a frame as base64 encoded string</p>
        <p><strong>Body:</strong> JSON with "frame" (base64 string)</p>
    </div>
    
    <div class="endpoint">
        <h3>POST /reset</h3>
        <p>Reset the frame buffer</p>
    </div>
    
    <div class="endpoint">
        <h3>GET /status</h3>
        <p>Check server status and buffer info</p>
    </div>
</body>
</html>
"""

def initialize_model(config_path):
    """Initialize the DeepPhys model for inference"""
    global model, device, config
    
    # Load config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=config_path, type=str)
    args = parser.parse_args(['--config_file', config_path])
    
    from main import add_args
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


def preprocess_frame(frame, config):
    """Preprocess a single frame for DeepPhys model"""
    # Resize
    h, w = config.TEST.DATA.PREPROCESS.RESIZE.H, config.TEST.DATA.PREPROCESS.RESIZE.W
    frame_resized = cv2.resize(frame, (w, h))
    
    # Normalize to [0, 1]
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    
    return frame_normalized


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
    
    # Ensure frames have 3 channels
    processed = []
    for frame in frames:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] != 3:
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
    predictions_np = predictions.cpu().numpy()
    
    return predictions_np


@app.route('/')
def index():
    """Home page with API documentation"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/status', methods=['GET'])
def status():
    """Get server status and buffer information"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'buffer_size': len(frame_buffer),
        'device': str(device) if device else 'not initialized',
        'chunk_size': chunk_size
    })


@app.route('/reset', methods=['POST'])
def reset_buffer():
    """Reset the frame buffer"""
    global frame_buffer
    frame_buffer = []
    return jsonify({'status': 'buffer_reset', 'buffer_size': 0})


@app.route('/infer_frame', methods=['POST'])
def infer_frame():
    """Receive a frame and return prediction"""
    global frame_buffer
    
    try:
        # Get frame from request
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        
        # Read image
        image_bytes = frame_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add to buffer
        frame_buffer.append(frame_rgb)
        
        # Process if we have enough frames
        if len(frame_buffer) >= chunk_size:
            # Get chunk
            chunk = np.array(frame_buffer[:chunk_size])
            
            # Run prediction
            predictions = predict_from_frames(chunk, config)
            
            # Remove processed frames from buffer
            frame_buffer = frame_buffer[chunk_size:]
            
            # Calculate mean prediction
            mean_pred = float(np.mean(predictions))
            
            return jsonify({
                'status': 'success',
                'prediction': mean_pred,
                'predictions': predictions.flatten().tolist(),
                'frames_processed': chunk_size,
                'buffer_remaining': len(frame_buffer)
            })
        else:
            return jsonify({
                'status': 'buffering',
                'message': f'Received {len(frame_buffer)}/{chunk_size} frames',
                'buffer_size': len(frame_buffer)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/infer_frame_base64', methods=['POST'])
def infer_frame_base64():
    """Receive frame as base64 encoded string"""
    global frame_buffer
    
    try:
        data = request.get_json()
        
        if 'frame' not in data:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Decode base64 image
        frame_data = data['frame']
        if frame_data.startswith('data:image'):
            # Remove data URL prefix
            frame_data = frame_data.split(',')[1]
        
        image_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add to buffer
        frame_buffer.append(frame_rgb)
        
        # Process if we have enough frames
        if len(frame_buffer) >= chunk_size:
            # Get chunk
            chunk = np.array(frame_buffer[:chunk_size])
            
            # Run prediction
            predictions = predict_from_frames(chunk, config)
            
            # Remove processed frames from buffer
            frame_buffer = frame_buffer[chunk_size:]
            
            # Calculate mean prediction
            mean_pred = float(np.mean(predictions))
            
            return jsonify({
                'status': 'success',
                'prediction': mean_pred,
                'predictions': predictions.flatten().tolist(),
                'frames_processed': chunk_size,
                'buffer_remaining': len(frame_buffer)
            })
        else:
            return jsonify({
                'status': 'buffering',
                'message': f'Received {len(frame_buffer)}/{chunk_size} frames',
                'buffer_size': len(frame_buffer)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import sys
    
    # Get config file path
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml'
    
    print("\n" + "=" * 60)
    print("Initializing rPPG Inference Server")
    print("=" * 60)
    
    # Initialize model
    model, config = initialize_model(config_path)
    
    print("\n" + "=" * 60)
    print("Starting Flask server on http://localhost:5000")
    print("=" * 60)
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

