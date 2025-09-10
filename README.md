COBRA Camera Vision System
This repository contains the camera vision system for a search robot, designed to stream real-time RGB and thermal video from a Raspberry Pi 5 to web and VR clients for disaster response scenarios. The repository is clonable for easy access and deployment, with two main folders:

raspberrypi5/: Python scripts for different streaming stages and corresponding HTML files for a web-based receiving client.
unity_client/: Codes for a Unity-based VR receiving client.

Clone the repository:
git clone https://github.com/priyankac16/Camera.git

Table of Contents

About the Project
Features
Repository Structure
Prerequisites
Installation
Usage
Auto-Start Setup
Contact

About the Project
We have a search robot for navigating complex terrains in disaster scenarios, such as collapsed buildings. This system uses WebRTC to stream video from a Raspberry Pi v2 Camera (RGB) and MLX90640 IR Array Thermal Camera, with YOLO for object detection. It supports two receiving clients:

A web-based client (in raspberrypi5/) for viewing streams in a browser.
A Unity-based VR client (in unity_client/) for immersive visualization.

Key files in raspberrypi5/ include:

camera_audio_stream.py: Streams RGB and thermal video and Audio.
camera_audio_stream_index.html: Web interface for the receiving client.
start_app.sh: Auto-runs the application on Raspberry Pi.
requirements.txt: Python dependencies (in root).

Features

Real-time RGB and thermal video streaming via WebRTC
Web interface (camera_audio_stream_index.html) for viewing live video and audio streaming with real time logs.
Thermal output with 20°C–45°C range, INFERNO colormap, and colorbar
YOLO object detection for victim identification
Optimized for low latency (320x240 resolution, 15 FPS)
Auto-start on Raspberry Pi boot
Unity-based VR client for immersive viewing

Repository Structure

raspberrypi5/:
Python scripts for different streaming stages ( only RGB camera, RGB+Thermal camera, RGB+Thermal+Audio) and corresponding HTML files for the web-based receiving client


unity_client/:
Python scripts that deal with Unity as the receiving client. The scripts according to their names, use a USB camera/ Intel realsense D455 and work with Nvidia's Jetson nano


requirements.txt: Python dependencies for raspberrypi5/ scripts
start_app.sh: Auto-start script for Raspberry Pi
README.md: This file

Prerequisites

Hardware:
Raspberry Pi 5 with Raspberry Pi OS (64-bit)
Raspberry Pi v2 Camera
MLX90640 IR Array Thermal Camera
Nvidia's Jetson nano
Intel realsense D455 Camera
USB Camera


Software (Raspberry Pi):
Python 3.10+ 
Node.js v16+
pip, npm, GStreamer (rpicamsrc), WebRTC-compatible browser (e.g., Chrome, Firefox)


Software (Windows development):
Python 3.10+, Git, text editor (e.g., VS Code)



Installation

Clone the repository:
git clone https://github.com/priyankac16/Camera.git

On Windows (PowerShell):
git clone https://github.com/priyankac16/Camera.git


Navigate to the project directory:
cd Camera

On Windows:
cd Camera

Note: Cloning creates a Camera folder.

Set up a virtual environment (in Camera/):
On Raspberry Pi:
python3.10 -m venv camera_env
source camera_env/bin/activate

On Windows:
python -m venv camera_env
.\camera_env\Scripts\activate


Install Python dependencies:
pip install -r requirements.txt

If issues arise, install manually:
pip install opencv-python aiortc websockets ultralytics numpy==1.26.4 adafruit-circuitpython-mlx90640 pyaudio av

On Raspberry Pi, ensure build tools:
sudo apt install libasound-dev portaudio19-dev libavcodec-dev libavformat-dev libswscale-dev python3-dev build-essential


Install Node.js dependencies (for WebRTC signaling):
npm install


Install GStreamer (Raspberry Pi):
sudo apt update
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-rpicamsrc


Enable Raspberry Pi camera:
sudo raspi-config

Go to Interface Options > Camera, enable, reboot:
sudo reboot


Configure MLX90640:
Connect to I2C pins. Enable I2C:
sudo raspi-config

Go to Interface Options > I2C, enable. Add user:
sudo usermod -a -G i2c rika


Set up environment variables (in Camera/):
Create .env:
touch .env

On Windows:
New-Item -ItemType File -Name .env

Add:
WEBRTC_PORT=8080
CAMERA_IP=192.168.1.100  # Replace with Raspberry Pi’s IP (run `hostname -I`)



Usage
Web Client (raspberrypi5/)

Activate virtual environment (in Camera/):
On Raspberry Pi:
source camera_env/bin/activate

On Windows:
.\camera_env\Scripts\activate


Start WebRTC server and streams:
python raspberrypi5/http.server 8000 

Start WebRTC server and streams:
python raspberrypi5/camera_audio_stream.py


Access web interface:
Open a browser:
http://<your-raspberry-pi-ip>:8080

Replace <your-raspberry-pi-ip> with the Raspberry Pi’s IP (e.g., 192.168.1.100).

View streams:
raspberrypi5/index.html displays:

"RGB Camera" (Raspberry Pi v2 Camera)
"Thermal Camera" (MLX90640, 20°C–45°C, INFERNO colormap with colorbar)
Audio Stream



Unity Client (unity_client/)

Set up Unity:

Open the unity_client/ folder in Unity.
Install a WebRTC package (e.g., Unity WebRTC) via Unity Package Manager.
Configure the project to connect to ws://<your-raspberry-pi-ip>:8765.


Run the Unity client:

Build and run the Unity project for your VR headset.
Ensure the Raspberry Pi is running raspberrypi5/cobrastream.py to stream video.



Auto-Start Setup
To auto-run on Raspberry Pi:

Verify start_app.sh (in Camera/):
Ensure it contains:
#!/bin/bash

# Navigate to your project directory

# Activate the virtual environment
source camera_env/bin/activate

# Run the web server in the background
python3 -m http.server 8000 &

# Run the WebSocket signaling server in the background

# Run your camera stream script
python3 camera_audio_stream.py

Make executable:
chmod +x start_app.sh


Set up systemd service:
Create /etc/systemd/system/camera_stream.service:
sudo nano /etc/systemd/system/camera_stream.service

Add:
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

Enable and start:
sudo systemctl enable camera_stream.service
sudo systemctl start camera_stream.service


Check status:
sudo systemctl status camera_stream.service

For audio issues, test mono audio:
arecord -D plughw:0,0 -f S16_LE -c1 -r16000 test.wav

Or disable audio in raspberrypi5/cobrastream.py.


Contact
Priyanka - priyankaachaudhary.1642003@gmail.comProject Link: https://github.com/priyankac16/Camera
