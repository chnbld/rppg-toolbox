#!/bin/bash
# Install WebSocket dependencies in the correct conda environment

echo "Installing WebSocket dependencies..."

# Install packages
pip install websockets opencv-python-headless --root-user-action=ignore

echo ""
echo "✓ Installation complete!"
echo ""
echo "To run the WebSocket server:"
echo "  python websocket_server.py configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml"

