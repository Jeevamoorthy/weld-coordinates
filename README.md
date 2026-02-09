MetalVision-Coordinate-Suite 🛠️📐
A specialized computer vision toolkit for detecting edges, corners, lines, and physical dimensions on metal surfaces. This suite is designed to handle the challenges of monochromatic (single-color) materials and reflective surfaces common in industrial environments.


![analyzed_2](https://github.com/user-attachments/assets/317f7530-d5a7-431a-a3a6-64e71c0748ec)


🌟 Key Features
Interactive Measurement (main.py): Select a Region of Interest (ROI) and calibrate pixels to millimeters using a real-world reference.
AI-Powered Segmentation (1.py): Leverages Meta's Segment Anything Model (SAM) for high-precision object masking.
Topological Corner Detection (2.py): Uses morphological gradients and Shi-Tomasi algorithms to find bends and welds on same-color surfaces.
Batch Line Processing (batch_process.py): Automatically extracts structural lines and start/end coordinates from entire folders of images.
⚙️ Installation
1. Clone the repository
code
Bash
git clone https://github.com/yourusername/MetalVision-Coordinate-Suite.git
cd MetalVision-Coordinate-Suite
2. Install Dependencies
code
Bash
pip install opencv-python numpy ultralytics
3. Download AI Weights
For 1.py, you need the Mobile SAM weights. The script will attempt to download them automatically via the Ultralytics API, or you can place mobile_sam.pt in the root directory.
🚀 Module Overview
1. Interactive Coordinator (main.py)
The flagship tool for real-world measurements.
Features: ROI selection, live Canny threshold tuning, and pixel-to-mm calibration.
Usage:
Draw a box over the metal part.
Press c to calibrate: click two points on a ruler in the frame and enter the distance in mm.
The system will now display coordinates in millimeters instead of pixels.
2. Live Corner Tuner (2.py)
Best for parts where edges are hard to see due to uniform lighting.
Algorithm: Uses a Bilateral Filter to preserve edges while removing surface noise (like scratches or rust) and a Morphological Gradient to highlight surface topology.
Controls: Use the UI sliders to adjust detection sensitivity in real-time.
3. Automated Line Extractor (batch_process.py)
Designed for batch processing of structural components (pipes, plates, beams).
Input: Place images in the /sample folder.
Output: Processed images with labeled line coordinates are saved to /output_lines.
4. SAM Segmentation (1.py)
A script to run Mobile SAM for zero-shot segmentation. Useful for creating masks of complex metal parts where traditional edge detection fails.
⌨️ Controls & Hotkeys
Key	Action
Mouse Drag	Define Region of Interest (ROI)
c	Start Calibration (Click 2 points on a known object)
r	Reset ROI and Calibration
q	Quit application
📁 Project Structure
code
Text
├── sample/               # Input folder for batch processing
├── output_lines/         # Output folder for processed images
├── 1.py                  # SAM Segmentation
├── 2.py                  # Real-time Corner Detection
├── batch_process.py      # Automated Line Extraction
├── main.py               # Main Interactive Calibration Tool
└── mobile_sam.pt         # (Optional) SAM Model Weights
🔧 Hardware Configuration
To switch between a local USB webcam and an IP-based camera, edit the WEBCAM_ID variable in the scripts:
code
Python
# For local USB camera
WEBCAM_ID = 0 

# For IP Camera (DroidCam/IP Webcam)
WEBCAM_ID = "https://192.168.1.XX:8080/video"
