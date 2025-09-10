# COBRA Camera Vision System

This repository contains the camera vision system for a search robot, designed to stream real-time RGB and thermal video from a Raspberry Pi 5 to web and VR clients for disaster response scenarios. The repository is clonable for easy access and deployment, with two main folders:

- **raspberrypi5/**: Python scripts for different streaming stages and corresponding HTML files for a web-based receiving client.
- **unity_client/**: Codes for a Unity-based VR receiving client.

Clone the repository:

```bash
git clone https://github.com/priyankac16/Camera.git
```

## Table of Contents

- About the Project
- Features
- Repository Structure
- Prerequisites
- Installation
- Usage
- Auto-Start Setup
- Contact

## About the Project

We have a search robot for navigating complex terrains in disaster scenarios, such as collapsed buildings. This system uses WebRTC to stream video from a Raspberry Pi v2 Camera (RGB) and MLX90640 IR Array Thermal Camera, with YOLO for object detection. It supports two receiving clients:

- A web-based client (in `raspberrypi5/`) for viewing streams in a browser.
- A Unity-based VR client (in `unity_client/`) for immersive visualization.

Key files in `raspberrypi5/` include:

- `camera_audio_stream.py`: Streams RGB and thermal video and Audio.
- `camera_audio_stream_index.html`: Web interface for the receiving client.
- `start_app.sh`: Auto-runs the application on Raspberry Pi.
- `requirements.txt`: Python dependencies (in root).

## Features

- Real-time RGB and thermal video streaming via WebRTC
- Web interface (`camera_audio_stream_index.html`) for viewing live video and audio streaming with real time logs.
- Thermal output with 20°C–45°C range, INFERNO colormap, and colorbar
- YOLO object detection for victim identification
- Optimized for low latency (320x240 resolution, 15 FPS)
- Auto-start on Raspberry Pi boot
- Unity-based VR client for immersive viewing

## Repository Structure

- `raspberrypi5/`:
  - Python scripts for different streaming stages ( only RGB camera, RGB+Thermal camera, RGB+Thermal+Audio) and corresponding HTML files for the web-based receiving client
- `unity_client/`:
  - Python scripts that deal with Unity as the receiving client. The scripts according to their names, use a USB camera/ Intel realsense D455 and work with Nvidia's Jetson nano
- `requirements.txt`: Python dependencies for `raspberrypi5/` scripts
- `start_app.sh`: Auto-start script for Raspberry Pi
- `README.md`: This file

## Prerequisites

- **Hardware**:
  - Raspberry Pi 5 with Raspberry Pi OS (64-bit)
  - Raspberry Pi v2 Camera
  - MLX90640 IR Array Thermal Camera
  - Nvidia's Jetson nano
  - Intel realsense D455 Camera
  - USB Camera
- **Software** (Raspberry Pi):
  - Python 3.10+ 
  - Node.js v16+
  - `pip`, `npm`, GStreamer (`rpicamsrc`), WebRTC-compatible browser (e.g., Chrome, Firefox)
- **Software** (Windows development):
  - Python 3.10+, Git, text editor (e.g., VS Code)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/priyankac16/Camera.git
   ```

   On Windows (PowerShell):

   ```powershell
   git clone https://github.com/priyankac16/Camera.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd Camera
   ```

   On Windows:

   ```powershell
   cd Camera
   ```

   *Note*: Cloning creates a `Camera` folder.

3. **Set up a virtual environment** (in `Camera/`):

   On Raspberry Pi:

   ```bash
   python3.10 -m venv camera_env
   source camera_env/bin/activate
   ```

   On Windows:

   ```powershell
   python -m venv camera_env
   .\camera_env\Scripts\activate
   ```

4. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   If issues arise, install manually:

   ```bash
   pip install opencv-python aiortc websockets ultralytics numpy==1.26.4 adafruit-circuitpython-mlx90640 pyaudio av
   ```

   On Raspberry Pi, ensure build tools:

   ```bash
   sudo apt install libasound-dev portaudio19-dev libavcodec-dev libavformat-dev libswscale-dev python3-dev build-essential
   ```

5. **Install Node.js dependencies** (for WebRTC signaling):

   ```bash
   npm install
   ```

6. **Install GStreamer** (Raspberry Pi):

   ```bash
   sudo apt update
   sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-rpicamsrc
   ```

7. **Enable Raspberry Pi camera**:

   ```bash
   sudo raspi-config
   ```

   Go to `Interface Options` &gt; `Camera`, enable, reboot:

   ```bash
   sudo reboot
   ```

8. **Configure MLX90640**:

   Connect to I2C pins. Enable I2C:

   ```bash
   sudo raspi-config
   ```

   Go to `Interface Options` &gt; `I2C`, enable. Add user:

   ```bash
   sudo usermod -a -G i2c rika
   ```

9. **Set up environment variables** (in `Camera/`):

   Create `.env`:

   ```bash
   touch .env
   ```

   On Windows:

   ```powershell
   New-Item -ItemType File -Name .env
   ```

   Add:

   ```bash
   WEBRTC_PORT=8080
   CAMERA_IP=192.168.1.100  # Replace with Raspberry Pi’s IP (run `hostname -I`)
   ```

## Usage

### Web Client (raspberrypi5/)

1. **Activate virtual environment** (in `Camera/`):

   On Raspberry Pi:

   ```bash
   source camera_env/bin/activate
   ```

   On Windows:

   ```powershell
   .\camera_env\Scripts\activate
   ```

2. **Start WebRTC server and streams**:

   ```bash
   python raspberrypi5/http.server 8000 
   ```

   **Start WebRTC server and streams**:

   ```bash
   python raspberrypi5/camera_audio_stream.py
   ```

3. **Access web interface**:

   Open a browser:

   ```bash
   http://<your-raspberry-pi-ip>:8080
   ```

   Replace `<your-raspberry-pi-ip>` with the Raspberry Pi’s IP (e.g., `192.168.1.100`).

4. **View streams**:

   `raspberrypi5/index.html` displays:

   - "RGB Camera" (Raspberry Pi v2 Camera)
   - "Thermal Camera" (MLX90640, 20°C–45°C, INFERNO colormap with colorbar)
   - Audio Stream

### Unity Client (unity_client/)

1. **Set up Unity**:

   - Open the `unity_client/` folder in Unity.
   - Install a WebRTC package (e.g., Unity WebRTC) via Unity Package Manager.
   - Configure the project to connect to `ws://<your-raspberry-pi-ip>:8765`.

2. **Run the Unity client**:

   - Build and run the Unity project for your VR headset.
   - Ensure the Raspberry Pi is running `raspberrypi5/cobrastream.py` to stream video.

## Auto-Start Setup

To auto-run on Raspberry Pi:

1. **Verify** `start_app.sh` (in `Camera/`):

   Ensure it contains:

   ```bash
   #!/bin/bash
   
   # Navigate to your project directory
   
   # Activate the virtual environment
   source camera_env/bin/activate
   
   # Run the web server in the background
   python3 -m http.server 8000 &
   
   # Run the WebSocket signaling server in the background
   
   # Run your camera stream script
   python3 camera_audio_stream.py
   ```

   Make executable:

   ```bash
   chmod +x start_app.sh
   ```

2. **Set up systemd service**:

   Create `/etc/systemd/system/camera_stream.service`:

   ```bash
   sudo nano /etc/systemd/system/camera_stream.service
   ```

   Add:

   ```bash
   [Unit]
   Description=COBRA Camera Stream Service
   After=network.target
   
   [Service]
   User=rika
   WorkingDirectory=/home/rika/Camera
   ExecStart=/home/rika/Camera/start_app.sh
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start:

   ```bash
   sudo systemctl enable camera_stream.service
   sudo systemctl start camera_stream.service
   ```

3. **Check status**:

   ```bash
   sudo systemctl status camera_stream.service
   ```

   For audio issues, test mono audio:

   ```bash
   arecord -D plughw:0,0 -f S16_LE -c1 -r16000 test.wav
   ```

   Or disable audio in `raspberrypi5/cobrastream.py`.

## Contact

Priyanka - priyankaachaudhary.1642003@gmail.com\
Project Link: https://github.com/priyankac16/Camera