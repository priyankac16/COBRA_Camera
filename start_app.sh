#!/bin/bash

# Navigate to your project directory
# This is a critical step to ensure relative paths work correctly.


# Activate the virtual environment.
# This ensures all subsequent python commands use the correct installed packages.
source camera_env/bin/activate

# Run the HTTP server in the background to serve index.html.
# The `&` symbol is crucial for running this process in the background.
python3 -m http.server 8000 &

# Run the WebSocket signaling server in the background.
# This script is required for WebRTC to establish a connection.


# Run the main camera stream script.
# This script handles the camera capture and WebRTC peer connection.
python3 camera_audio_stream.py
